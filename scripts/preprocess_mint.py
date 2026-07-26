#!/usr/bin/env python3
"""Preprocess HumanML3D → MinT OpenSim q + MinT muscle labels (NPZ cache).

Run from the SINDyffuse repo root, e.g.::

  python scripts/preprocess_mint.py --max_motions 1
  python scripts/preprocess_mint.py --skip_existing --num_workers 8
  python scripts/preprocess_mint.py --num_shards 128 --shard_index 0 --skip_normalization
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from concurrent.futures.process import BrokenProcessPool
except ImportError:
    BrokenProcessPool = BrokenExecutor  # type: ignore[misc, assignment]

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

warnings.filterwarnings(
    "ignore",
    message=r"A NumPy version.*SciPy",
    category=UserWarning,
)

from common.cpu import detect_usable_cpus, resolve_k8s_shard, resolve_preprocess_parallelism
from common.paths import default_humanml3d_root, default_mint_root, mint_cache_dir
from common.run_logging import (
    RunLogger,
    add_run_log_cli_args,
    dual_tqdm,
    null_logger,
    run_log_session,
)
from common.run_setup import apply_preprocess_job_env
from common.skeleton_config import DEFAULT_FPS
from datasets.hml3d_joints import default_joints_root, load_hml3d_joint_positions
from datasets.mint_cache_stats import compute_mint_normalization_stats
from datasets.splits import all_motion_ids, shard_motion_ids
from mint.cache_schema import write_cache_metadata, write_motion_cache
from mint.features import features_from_mint_q
from mint.label_lookup import lookup_hml_motion
from mint.physics import bio_matrix_mint
from mint.retarget import retarget_hml_joints_to_q

if TYPE_CHECKING:
    WorkItem = tuple[
        str,
        str,
        str,
        str,
        float,
        bool,
        str,
        str,
        str,
    ]

_WORKER_OPENSIM_LOG_LEVEL = "Off"


def _check_runtime_dependencies() -> None:
    """Fail fast when required MinT preprocess deps are missing."""
    import scipy

    major_minor = tuple(int(x) for x in scipy.__version__.split(".")[:2])
    if major_minor >= (1, 14):
        raise RuntimeError(
            f"SciPy {scipy.__version__} requires numpy>=1.26; OpenSim needs numpy 1.25.x. "
            "Rebuild: mamba env update -f env/environment.yaml --prune"
        )
    try:
        import musint  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "musint is required for MinT muscle labels. "
            "Install: pip install 'musint>=0.3.0' (or mamba env update -f env/environment.yaml)"
        ) from exc


def _mint_worker_init(log_level: str) -> None:
    from nimble.muscle_activation import configure_opensim_logging

    global _WORKER_OPENSIM_LOG_LEVEL
    _WORKER_OPENSIM_LOG_LEVEL = str(log_level)
    configure_opensim_logging(log_level)


def export_motion_to_npz(
    motion_id: str,
    *,
    hml_root: Path,
    out_root: Path,
    mint_root: str,
    fps: float = DEFAULT_FPS,
    skip_existing: bool = False,
    joint_source: str = "auto",
    joints_root: str = "",
) -> dict:
    out_path = mint_cache_dir(out_root) / f"{motion_id}.npz"
    if skip_existing and out_path.is_file():
        return {"id": motion_id, "status": "skipped", "path": str(out_path)}

    try:
        joints, _ = load_hml3d_joint_positions(
            hml_root,
            motion_id,
            joint_source=joint_source,  # type: ignore[arg-type]
            joints_root=joints_root or None,
        )
        rt = retarget_hml_joints_to_q(joints, fps=fps)
        q = rt.q
        t_len = int(q.shape[0])

        label = lookup_hml_motion(
            motion_id,
            mint_root=mint_root,
            num_frames=t_len,
            target_fps=fps,
        )
        act = label.activations
        if act.shape[0] != t_len:
            if act.shape[0] > t_len:
                act = act[:t_len]
            else:
                pad = np.zeros((t_len - act.shape[0], act.shape[1]), dtype=np.float32)
                act = np.concatenate([act, pad], axis=0)

        bio = bio_matrix_mint(q, fps=fps)
        u, c, _, _ = features_from_mint_q(q, fps=fps)
        sindy_feat = np.concatenate([u, c], axis=-1).astype(np.float32)

        write_motion_cache(
            out_path,
            q=q,
            muscle_activations=act,
            guidance_features=bio,
            sindy_features=sindy_feat,
            has_mint_labels=label.has_labels,
            fps=fps,
        )
        return {
            "id": motion_id,
            "status": "ok",
            "path": str(out_path),
            "num_frames": t_len,
            "num_dofs": int(q.shape[1]),
            "has_mint_labels": bool(label.has_labels),
            "mean_fk_error": float(rt.mean_fk_error),
            "retarget_method": str(getattr(rt, "method", "")),
        }
    except Exception as exc:
        return {"id": motion_id, "status": "error", "error": str(exc)}


def _process_one_mint(args: WorkItem) -> dict:
    (
        sid,
        hml_root_s,
        out_root_s,
        mint_root_s,
        fps,
        skip_existing,
        joint_source,
        joints_root_s,
        verbose_log_path,
    ) = args
    from nimble.muscle_activation import configure_opensim_logging

    configure_opensim_logging(_WORKER_OPENSIM_LOG_LEVEL)
    if verbose_log_path:
        os.environ["SINDYFFUSE_VERBOSE_LOG"] = str(verbose_log_path)
    else:
        os.environ.pop("SINDYFFUSE_VERBOSE_LOG", None)

    return export_motion_to_npz(
        sid,
        hml_root=Path(hml_root_s),
        out_root=Path(out_root_s),
        mint_root=mint_root_s,
        fps=float(fps),
        skip_existing=bool(skip_existing),
        joint_source=str(joint_source),
        joints_root=str(joints_root_s),
    )


def _manifest_path(out_root: Path, shard_index: int, num_shards: int) -> Path:
    cache = mint_cache_dir(out_root)
    if int(num_shards) > 1:
        return cache / f"preprocess_mint_manifest.{int(shard_index):04d}.jsonl"
    return cache / "preprocess_mint_manifest.jsonl"


def run_preprocess(args: argparse.Namespace, logger: RunLogger | None = None) -> None:
    log = logger or null_logger()
    _check_runtime_dependencies()

    hml_root = Path(args.hml_root or default_humanml3d_root()).expanduser().resolve()
    out_root = Path(args.out_root or default_humanml3d_root()).expanduser().resolve()
    mint_root = str(args.mint_root or default_mint_root())
    cache = mint_cache_dir(out_root)
    cache.mkdir(parents=True, exist_ok=True)

    ids = all_motion_ids(hml_root)
    if int(args.max_motions) > 0:
        ids = ids[: int(args.max_motions)]

    cli_shards = getattr(args, "num_shards", None)
    cli_shard_index = getattr(args, "shard_index", None)
    shard_index, num_shards = resolve_k8s_shard(
        num_shards=int(cli_shards) if cli_shards is not None else None,
        shard_index=int(cli_shard_index) if cli_shard_index is not None else None,
    )
    if num_shards > 1:
        ids = shard_motion_ids(ids, shard_index, num_shards)
        log.progress(f"Shard {shard_index + 1}/{num_shards}: {len(ids)} motion(s)")

    joints_root = str(args.joints_root or "").strip()
    if not joints_root:
        jr = default_joints_root(hml_root)
        joints_root = str(jr) if jr is not None else ""

    fps = float(args.fps)
    skip_existing = bool(getattr(args, "skip_existing", False))
    joint_source = str(getattr(args, "joint_source", "auto"))
    verbose_log_path = str(getattr(args, "_run_log_file", "") or "").strip()
    log_level = str(getattr(args, "opensim_log_level", "Off"))

    work: list[WorkItem] = [
        (
            sid,
            str(hml_root),
            str(out_root),
            mint_root,
            fps,
            skip_existing,
            joint_source,
            joints_root,
            verbose_log_path,
        )
        for sid in ids
    ]

    ok = err = skip = 0
    num_dofs: int | None = None
    total_motions = len(work)
    if total_motions == 0:
        log.progress("No motions to process; exiting successfully")
        return

    motion_workers, _ = resolve_preprocess_parallelism(
        int(getattr(args, "num_workers", 0)),
        activation_method="none",
        skip_muscle_activation=True,
        num_shards=num_shards,
    )
    if num_shards > 1:
        log.progress(
            f"Distributed preprocess shard {shard_index}/{num_shards}: "
            f"{total_motions} motion(s), 1 worker per pod"
        )
    elif motion_workers > 1:
        log.progress(
            f"MinT preprocess: {motion_workers} parallel motion worker(s) "
            f"(usable_cpus={detect_usable_cpus()})"
        )

    skip_normalization = bool(getattr(args, "skip_normalization", False)) or num_shards > 1
    manifest_path = _manifest_path(out_root, shard_index, num_shards)
    pending = list(work)

    def _record(row: dict, *, pbar: dual_tqdm | None = None) -> None:
        nonlocal ok, err, skip, num_dofs
        status = row.get("status")
        if status == "ok":
            ok += 1
            nd = int(row.get("num_dofs", 0))
            if num_dofs is None:
                num_dofs = nd
            elif nd != num_dofs:
                log.verbose(f"WARNING: num_dofs mismatch {row['id']}: {nd} vs {num_dofs}")
        elif status == "skipped":
            skip += 1
        else:
            err += 1
            log.verbose(f"ERROR {row.get('id')}: {row.get('error')}")
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                {"last": str(row.get("id", "")), "status": str(row.get("status", ""))},
                refresh=False,
            )

    _mint_worker_init(log_level)
    while pending:
        if motion_workers <= 1:
            batch = pending
            pending = []
            with dual_tqdm(total=len(batch), desc="preprocess_mint", logger=log) as pbar:
                for w in batch:
                    row = _process_one_mint(w)
                    _record(row, pbar=pbar)
        else:
            batch = pending
            pending = []
            pool_kwargs: dict = {
                "max_workers": motion_workers,
                "mp_context": get_context("spawn"),
                "initializer": _mint_worker_init,
                "initargs": (log_level,),
            }
            if sys.version_info >= (3, 11):
                pool_kwargs["max_tasks_per_child"] = 1
            finished_in_batch: set[str] = set()
            pool_broken = False
            with dual_tqdm(total=len(batch), desc="preprocess_mint", logger=log) as pbar:
                try:
                    with ProcessPoolExecutor(**pool_kwargs) as ex:
                        futs = {ex.submit(_process_one_mint, w): w for w in batch}
                        for fut in as_completed(futs):
                            w = futs[fut]
                            try:
                                row = fut.result()
                            except (BrokenProcessPool, BrokenExecutor) as exc:
                                pool_broken = True
                                row = {
                                    "id": w[0],
                                    "status": "error",
                                    "error": (
                                        f"worker crashed (OpenSim/native): {exc}. "
                                        "Retrying remaining motions sequentially."
                                    ),
                                }
                            _record(row, pbar=pbar)
                            finished_in_batch.add(w[0])
                            if pool_broken:
                                break
                except (BrokenProcessPool, BrokenExecutor):
                    pool_broken = True

            if pool_broken:
                retry = [ww for ww in batch if ww[0] not in finished_in_batch]
                pending = retry + pending
                motion_workers = 1
                log.progress(
                    f"WARNING: process pool crashed; retrying {len(retry)} motion(s) sequentially"
                )

    with manifest_path.open("w", encoding="utf-8") as mf:
        mf.write(
            json.dumps(
                {
                    "ok": ok,
                    "skip": skip,
                    "err": err,
                    "num_dofs": num_dofs,
                    "num_shards": num_shards if num_shards > 1 else None,
                    "shard_index": shard_index if num_shards > 1 else None,
                    "motion_workers": motion_workers,
                }
            )
            + "\n"
        )

    log.progress(f"Done: ok={ok} skip={skip} err={err}")
    if num_dofs is not None:
        write_cache_metadata(cache, ndof=int(num_dofs), fps=fps)

    if not skip_normalization and ok > 0:
        stats = compute_mint_normalization_stats(out_root, split="train")
        log.progress(f"Wrote normalization: {stats}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess HumanML3D into MinT NPZ cache")
    parser.add_argument("--hml_root", default="", help="HumanML3D root")
    parser.add_argument("--out_root", default="", help="Output root (default: hml_root)")
    parser.add_argument("--mint_root", default="", help="MinT dataset root (MINT_ROOT)")
    parser.add_argument("--max_motions", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--skip_normalization", action="store_true")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--joint_source", default="auto")
    parser.add_argument("--joints_root", default="")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Parallel motion workers (0 = auto from CPU count; forced to 1 when sharded)",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split motions across N shards for distributed K8s runs (default 1 = no sharding)",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=-1,
        help="This shard index in [0, num_shards); default auto from JOB_COMPLETION_INDEX when sharded",
    )
    parser.add_argument(
        "--opensim_log_level",
        default="Off",
        help="OpenSim console log level in workers (default Off)",
    )
    add_run_log_cli_args(parser)
    args = parser.parse_args()
    apply_preprocess_job_env(args)

    shard_idx = int(args.shard_index)
    if args.num_shards > 1 and shard_idx < 0:
        args.shard_index = None
    elif shard_idx >= 0:
        args.shard_index = shard_idx

    if args.no_run_log:
        run_preprocess(args, null_logger())
        return

    num_shards = int(getattr(args, "num_shards", 1) or 1)
    shard_index = getattr(args, "shard_index", None)
    line_prefix = ""
    if num_shards > 1 and shard_index is not None and int(shard_index) >= 0:
        line_prefix = f"[shard={int(shard_index):04d}] "
        os.environ["SINDYFFUSE_VERBOSE_LOG_PREFIX"] = line_prefix.strip()

    run_log_id = str(getattr(args, "run_log_id", "") or "").strip() or None
    script_name = Path(__file__).stem
    with run_log_session(
        args.log_dir,
        script_name=script_name,
        argv=sys.argv,
        run_id=run_log_id,
        line_prefix=line_prefix,
    ) as (paths, logger):
        args._run_log_file = str(paths.log_file)
        logger.progress(f"log: {paths.latest_log}")
        run_preprocess(args, logger)


if __name__ == "__main__":
    main()
