#!/usr/bin/env python3
"""Preprocess HumanML3D motion into Nimble B3D cache (one .b3d per motion).

Run from the SINDyffuse repo root, e.g.:

  python scripts/preprocess_nimble.py
  python scripts/preprocess_nimble.py --num_workers 32   # Moco uses 32 threads (default: auto)
  python scripts/preprocess_nimble.py --skip_existing   # skip motions that already have .b3d
  python scripts/preprocess_nimble.py --moco_parallel_motions 4 --num_workers 64
  python scripts/preprocess_nimble.py --skip_muscle_activation --num_workers 8   # IK-only, 8 parallel
  python scripts/preprocess_nimble.py --max_motions 1   # smoke test
  python scripts/preprocess_nimble.py --num_shards 4 --shard_index 0 --skip_normalization  # K8s shard (local test)

Defaults: ``datasets/HumanML3D`` for input and output; writes ``nimble_b3d/*.b3d``
and q-space Mean.npy / Std.npy under that subfolder. The bundled Rajagopal 2015
skeleton is the only model -- see ``nimble/skeleton_registry.py`` for the IK
mapping.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import warnings
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import get_context

try:
    from concurrent.futures.process import BrokenProcessPool
except ImportError:
    BrokenProcessPool = BrokenExecutor  # type: ignore[misc, assignment]
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

warnings.filterwarnings(
    "ignore",
    message=r"A NumPy version.*SciPy",
    category=UserWarning,
)

# ``nimble.export`` is imported lazily inside workers to avoid eagerly loading
# the nimble skeleton in the parent process. ``compute_nimble_normalization_stats``
# only touches written B3D files (post-processing) so it's safe to import here.
from common.cpu import (
    configure_compute_threads,
    detect_usable_cpus,
    resolve_k8s_shard,
    resolve_preprocess_parallelism,
)
from common.paths import (
    NIMBLE_B3D_SUBDIR,
    cleanup_preprocess_manifests,
    default_humanml3d_root,
    humanml3d_text_dir,
    nimble_b3d_dir,
)
from common.run_logging import (
    DualTqdm,
    RunLogger,
    add_run_log_cli_args,
    append_verbose_log,
    dual_tqdm,
    null_logger,
    run_log_session,
)
from datasets.hml3d_joints import default_joints_root, load_hml3d_joint_positions
from datasets.nimble_dataset import compute_nimble_normalization_stats
from datasets.splits import all_motion_ids, shard_motion_ids
from nimble.muscle_activation import (
    add_muscle_activation_cli_args,
    configure_opensim_logging,
    muscle_activation_config_from_args,
    muscle_activation_config_from_dict,
    muscle_activation_config_to_dict,
    normalize_activation_method,
    resolve_activation_method,
)


WorkItem = tuple[
    str,
    str,
    str,
    float,
    float,
    float,
    bool,
    str,
    str,
    str,
    bool,
    str,
]

# Set in pool initializer so each worker suppresses OpenSim spam before loading models.
_WORKER_OPENSIM_LOG_LEVEL = "Off"
_WORKER_MOCO_THREADS = 1


def _preprocess_worker_init(log_level: str, moco_threads: int) -> None:
    global _WORKER_OPENSIM_LOG_LEVEL, _WORKER_MOCO_THREADS
    _WORKER_OPENSIM_LOG_LEVEL = str(log_level)
    _WORKER_MOCO_THREADS = int(moco_threads)
    configure_compute_threads(_WORKER_MOCO_THREADS)
    try:
        from nimble.muscle_activation import configure_opensim_logging

        configure_opensim_logging(log_level)
    except Exception:
        pass


def _load_joints(
    hml_root: Path,
    sid: str,
    *,
    joint_source: str,
    joints_root: Path | None,
) -> np.ndarray | None:
    try:
        joints, _ = load_hml3d_joint_positions(
            hml_root,
            sid,
            joint_source=joint_source,  # type: ignore[arg-type]
            joints_root=joints_root,
        )
        return joints
    except FileNotFoundError:
        return None


def _link_or_copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    if sys.platform != "win32":
        try:
            os.symlink(src, dst, target_is_directory=True)
            return
        except OSError:
            pass
    shutil.copytree(src, dst)


def _process_one(args: WorkItem) -> dict:
    (
        sid,
        hml_root_s,
        out_root_s,
        fps,
        mass_kg,
        height_m,
        skip_existing,
        joint_source,
        joints_root_s,
        act_cfg_json,
        skip_muscle_activation,
        verbose_log_path,
    ) = args
    from nimble.muscle_activation import configure_opensim_logging, opensim_quiet

    configure_opensim_logging(_WORKER_OPENSIM_LOG_LEVEL)
    if verbose_log_path:
        os.environ["SINDYFFUSE_VERBOSE_LOG"] = str(verbose_log_path)
    else:
        os.environ.pop("SINDYFFUSE_VERBOSE_LOG", None)

    hml_root = Path(hml_root_s)
    out_root = Path(out_root_s)
    joints_root = Path(joints_root_s) if joints_root_s else None
    out_b3d = nimble_b3d_dir(out_root) / f"{sid}.b3d"
    if skip_existing and out_b3d.is_file():
        append_verbose_log(f"{sid}: skipped (existing b3d)")
        return {"id": sid, "status": "skipped", "path": str(out_b3d)}

    joints = _load_joints(
        hml_root,
        sid,
        joint_source=joint_source,
        joints_root=joints_root,
    )
    if joints is None:
        append_verbose_log(f"{sid}: error missing or invalid motion")
        return {"id": sid, "status": "error", "error": "missing or invalid motion"}

    act_cfg = muscle_activation_config_from_dict(json.loads(act_cfg_json))
    append_verbose_log(
        f"{sid}: begin ({int(joints.shape[0])} frames, activation={act_cfg.activation_method})"
    )
    try:
        with opensim_quiet(act_cfg.opensim_log_level):
            from nimble.export import export_motion_to_b3d

            stats, num_dofs, meta_strings = export_motion_to_b3d(
                joints,
                out_b3d,
                trial_name=sid,
                fps=fps,
                mass_kg=mass_kg,
                height_m=height_m,
                muscle_activation_cfg=act_cfg,
                skip_muscle_activation=bool(skip_muscle_activation),
                activation_method=str(act_cfg.activation_method),
            )
    except Exception as exc:
        append_verbose_log(f"{sid}: error {exc}")
        return {"id": sid, "status": "error", "error": str(exc)}

    from nimble.export import clear_export_caches

    clear_export_caches()

    row: dict = {
        "id": sid,
        "status": "ok",
        "path": str(out_b3d),
        "num_dofs": int(num_dofs),
        "ik_stats": stats,
    }
    if meta_strings:
        row["meta"] = meta_strings
    return row


def _process_one_in_fresh_process(
    work_item: WorkItem,
    *,
    log_level: str,
    moco_threads: int,
) -> dict:
    """Run one motion in a short-lived worker process.

    OpenSim static optimization can retain native memory in long-lived Python
    processes. Uses ``spawn`` so the child does not inherit the parent's loaded
    OpenSim / Nimble / Torch heap from ``fork``.
    """
    ctx = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=ctx,
        initializer=_preprocess_worker_init,
        initargs=(log_level, moco_threads),
    ) as ex:
        return ex.submit(_process_one, work_item).result()


def _run_motion(
    work_item: WorkItem,
    *,
    isolate_motion_process: bool,
    log_level: str,
    moco_threads: int,
) -> dict:
    if isolate_motion_process:
        return _process_one_in_fresh_process(
            work_item,
            log_level=log_level,
            moco_threads=moco_threads,
        )
    return _process_one(work_item)


def _ik_fit_loss(ik_stats: dict) -> float | None:
    """Mean Nimble IK joint fitting loss (HumanML3D joints → Rajagopal ``q``)."""
    loss = ik_stats.get("mean_fit_joints_loss", ik_stats.get("mean_ik_error"))
    if loss is None:
        loss = ik_stats.get("mean_fk_loss")
    if loss is None:
        return None
    val = float(loss)
    return val if np.isfinite(val) else None


def _moco_activation_objective(ik_stats: dict) -> float | None:
    """Mean MocoTrack optimal-control objective across segments."""
    if not ik_stats.get("muscle_activation_computed"):
        return None
    for key in ("muscle_activation_objective", "moco_objective", "moco_moco_objective"):
        val = ik_stats.get(key)
        if val is None:
            continue
        out = float(val)
        return out if np.isfinite(out) else None
    return None


def _log_motion_verbose(row: dict, logger: RunLogger) -> None:
    """Per-motion summary (IK loss, activation objective, timing)."""
    mid = str(row.get("id", ""))
    status = row.get("status")
    logger.verbose(f"=== motion {mid} status={status} ===")

    if status == "skipped":
        reason = row.get("skip_reason", "existing")
        logger.verbose(f"{mid}: skipped ({reason})")
        return

    if status != "ok":
        logger.verbose(f"{mid}: error {row.get('error', 'unknown error')}")
        return

    ik = row.get("ik_stats") or {}
    ik_loss = _ik_fit_loss(ik)
    if ik_loss is not None:
        logger.verbose(f"{mid}: IK fit loss {ik_loss:.6f}")
    fk = ik.get("mean_fk_loss")
    if fk is not None and np.isfinite(float(fk)):
        logger.verbose(f"{mid}: IK mean_fk_loss {float(fk):.6f}")
    ratio = ik.get("success_ratio")
    if ratio is not None:
        logger.verbose(
            f"{mid}: IK success_ratio {float(ratio):.4f} "
            f"({int(ik.get('success_count', 0))}/{int(ik.get('total_frames', 0))} frames)"
        )
    act_sec = ik.get("muscle_activation_seconds")
    if act_sec is not None:
        logger.verbose(f"{mid}: muscle_activation_seconds {float(act_sec):.1f}")
    moco_obj = _moco_activation_objective(ik)
    if moco_obj is not None:
        logger.verbose(f"{mid}: muscle_activation_objective {moco_obj:.6f}")
    repaired = ik.get("muscle_activation_repaired_frames")
    if repaired is not None:
        logger.verbose(f"{mid}: muscle_activation_repaired_frames {int(repaired)}")
    for key in (
        "moco_solver_success",
        "moco_solver_status",
        "moco_solver_iterations",
        "moco_objective",
    ):
        val = ik.get(key)
        if val is not None:
            logger.verbose(f"{mid}: {key} {val}")


def _print_motion_progress(row: dict, *, logger: RunLogger) -> None:
    """Log per-motion metrics and warnings for IK / muscle-activation failures."""
    mid = str(row.get("id", ""))
    status = row.get("status")
    _log_motion_verbose(row, logger)

    if status == "skipped":
        return

    if status != "ok":
        err = row.get("error", "unknown error")
        if "IK" in str(err) or "ik" in str(err).lower():
            logger.warn(f"{mid} IK failed: {err}")
        elif "Muscle activation" in str(err) or "moco" in str(err).lower():
            logger.warn(f"{mid} muscle activation failed: {err}")
        else:
            logger.warn(f"{mid} failed: {err}")
        return

    ik = row.get("ik_stats") or {}
    min_success = 2
    success_count = int(ik.get("success_count", 0))
    if success_count < min_success:
        logger.warn(
            f"{mid} IK failed: success {success_count}/"
            f"{int(ik.get('total_frames', 0))}"
        )
    repaired = ik.get("muscle_activation_repaired_frames")
    if repaired is not None and float(repaired) > 0.0:
        logger.warn(f"{mid} muscle activation interpolated {int(repaired)} frames")


def _motion_loss(row: dict) -> float | None:
    if row.get("status") != "ok":
        return None
    return _ik_fit_loss(row.get("ik_stats") or {})


def _symlink_metadata(hml_root: Path, out_root: Path) -> None:
    if hml_root.resolve() == out_root.resolve():
        return
    src_texts = humanml3d_text_dir(hml_root)
    if src_texts.is_dir() and src_texts != hml_root.resolve():
        _link_or_copy_tree(src_texts, out_root / "texts")
    for name in ("train.txt", "val.txt", "test.txt"):
        src = hml_root / name
        dst = out_root / name
        if src.is_file() and not dst.exists():
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)


def _manifest_path(out_root: Path, shard_index: int, num_shards: int) -> Path:
    if int(num_shards) > 1:
        return out_root / f"preprocess_manifest.{int(shard_index):04d}.jsonl"
    return out_root / "preprocess_manifest.jsonl"


def run_preprocess(args: argparse.Namespace, logger: RunLogger | None = None) -> None:
    """Run preprocessing from a populated argparse namespace."""
    log = logger or null_logger()
    default_root = default_humanml3d_root()
    if int(np.__version__.split(".", maxsplit=1)[0]) >= 2:
        log.progress("ERROR: nimblephysics requires numpy<2 for stable IK")
        sys.exit(1)

    hml_root = Path(getattr(args, "hml_root", default_root) or default_root).expanduser().resolve()
    out_root = Path(getattr(args, "out_root", default_root) or default_root).expanduser().resolve()
    b3d_cache = nimble_b3d_dir(out_root)
    b3d_cache.mkdir(parents=True, exist_ok=True)

    ids = all_motion_ids(hml_root)
    max_motions = int(getattr(args, "max_motions", 0) or 0)
    if max_motions > 0:
        ids = ids[:max_motions]

    cli_shards = getattr(args, "num_shards", None)
    cli_shard_index = getattr(args, "shard_index", None)
    shard_index, num_shards = resolve_k8s_shard(
        num_shards=int(cli_shards) if cli_shards is not None else None,
        shard_index=int(cli_shard_index) if cli_shard_index is not None else None,
    )
    if num_shards > 1:
        ids = shard_motion_ids(ids, shard_index, num_shards)
        log.progress(
            f"Shard {shard_index + 1}/{num_shards}: {len(ids)} motion(s) assigned"
        )

    _symlink_metadata(hml_root, out_root)

    joints_root = str(getattr(args, "joints_root", "") or "").strip()
    if not joints_root:
        jr = default_joints_root(hml_root)
        joints_root = str(jr) if jr is not None else ""

    fps = float(getattr(args, "fps", 20.0))
    mass_kg = float(getattr(args, "mass_kg", 70.0))
    height_m = float(getattr(args, "height_m", 1.75))
    skip_existing = bool(getattr(args, "skip_existing", False))

    act_cfg = muscle_activation_config_from_args(args, fps=fps, mass_kg=mass_kg)
    act_cfg_json = json.dumps(muscle_activation_config_to_dict(act_cfg))
    activation_method = normalize_activation_method(act_cfg.activation_method)
    skip_muscle_activation = activation_method == "none"
    verbose_log_path = str(getattr(args, "_run_log_file", "") or "").strip()

    work: list[WorkItem] = [
        (
            sid,
            str(hml_root),
            str(out_root),
            fps,
            mass_kg,
            height_m,
            skip_existing,
            str(getattr(args, "joint_source", "auto")),
            joints_root,
            act_cfg_json,
            skip_muscle_activation,
            verbose_log_path,
        )
        for sid in ids
    ]

    ok = err = skip = 0
    num_dofs_ref: int | None = None
    loss_sum = 0.0
    loss_count = 0
    moco_obj_sum = 0.0
    moco_obj_count = 0
    total_motions = len(work)
    if total_motions == 0:
        log.progress("No motions to process; exiting successfully")
        sys.exit(0)

    def _record(row: dict, *, pbar: DualTqdm | None = None) -> None:
        nonlocal ok, err, skip, num_dofs_ref, loss_sum, loss_count
        nonlocal moco_obj_sum, moco_obj_count
        _print_motion_progress(row, logger=log)
        loss = _motion_loss(row)
        status = row.get("status")
        if status == "ok":
            ok += 1
            if loss is not None:
                loss_sum += loss
                loss_count += 1
            ik = row.get("ik_stats") or {}
            moco_obj = _moco_activation_objective(ik)
            if moco_obj is not None:
                moco_obj_sum += moco_obj
                moco_obj_count += 1
            nd = int(row.get("num_dofs", 0))
            if num_dofs_ref is None:
                num_dofs_ref = nd
            elif nd != num_dofs_ref:
                log.verbose(
                    f"WARNING: num_dofs mismatch {row['id']}: {nd} vs {num_dofs_ref}"
                )
        elif status == "skipped":
            skip += 1
        else:
            err += 1
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                {"last": str(row.get("id", "")), "status": str(row.get("status", ""))},
                refresh=False,
            )

    log_level = str(getattr(args, "opensim_log_level", "Off"))
    configure_opensim_logging(log_level)
    moco_parallel = int(getattr(args, "moco_parallel_motions", 1) or 1)
    motion_workers, moco_threads = resolve_preprocess_parallelism(
        int(getattr(args, "num_workers", 0)),
        activation_method=activation_method,
        skip_muscle_activation=skip_muscle_activation,
        moco_parallel_motions=moco_parallel,
        num_shards=num_shards,
    )
    if activation_method != "none":
        configure_compute_threads(moco_threads)
    if num_shards > 1:
        log.verbose(
            f"Distributed preprocess shard {shard_index}/{num_shards}: "
            f"1 in-pod worker, {moco_threads} OpenSim thread(s)"
        )
    elif activation_method == "none":
        log.verbose(
            f"IK-only preprocess: {motion_workers} parallel motion worker(s) "
            f"(usable_cpus={detect_usable_cpus()}, activation_method=none)"
        )
    elif activation_method == "static_optimization":
        log.verbose(
            f"Static-opt preprocess: {motion_workers} parallel motion worker(s) "
            f"(usable_cpus={detect_usable_cpus()})"
        )
    elif motion_workers > 1:
        log.verbose(
            f"Moco preprocess: {motion_workers} motion(s) in parallel, "
            f"{moco_threads} thread(s) each "
            f"(usable_cpus={detect_usable_cpus()})"
        )
    else:
        log.verbose(
            f"Moco preprocess: 1 motion at a time, {moco_threads} thread(s) "
            f"(usable_cpus={detect_usable_cpus()})"
        )

    manifest_path = _manifest_path(out_root, shard_index, num_shards)
    skip_normalization = bool(getattr(args, "skip_normalization", False)) or num_shards > 1
    isolate_motion_process = bool(activation_method == "static_optimization")
    pending = list(work)
    with manifest_path.open("w", encoding="utf-8") as mf:
        while pending:
            if motion_workers <= 1:
                batch = pending
                pending = []
                pbar = dual_tqdm(
                    total=len(batch),
                    desc="preprocess",
                    unit="motion",
                    logger=log,
                )
                try:
                    for w in batch:
                        row = _run_motion(
                            w,
                            isolate_motion_process=isolate_motion_process,
                            log_level=log_level,
                            moco_threads=1 if isolate_motion_process else moco_threads,
                        )
                        mf.write(json.dumps(row, default=str) + "\n")
                        mf.flush()
                        os.fsync(mf.fileno())
                        _record(row, pbar=pbar)
                finally:
                    pbar.close()
            else:
                batch = pending
                pending = []
                pool_kwargs: dict = {
                    "max_workers": motion_workers,
                    "initializer": _preprocess_worker_init,
                    "initargs": (log_level, moco_threads if not skip_muscle_activation else 1),
                }
                if sys.version_info >= (3, 11):
                    pool_kwargs["max_tasks_per_child"] = 1
                finished_in_batch: set[str] = set()
                pool_broken = False
                pbar = dual_tqdm(
                    total=len(batch),
                    desc="preprocess",
                    unit="motion",
                    logger=log,
                )
                try:
                    with ProcessPoolExecutor(**pool_kwargs) as ex:
                        futs = {
                            ex.submit(
                                _run_motion,
                                w,
                                isolate_motion_process=isolate_motion_process,
                                log_level=log_level,
                                moco_threads=1 if isolate_motion_process else moco_threads,
                            ): w
                            for w in batch
                        }
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
                            mf.write(json.dumps(row, default=str) + "\n")
                            mf.flush()
                            os.fsync(mf.fileno())
                            _record(row, pbar=pbar)
                            finished_in_batch.add(w[0])
                            if pool_broken:
                                break
                except (BrokenProcessPool, BrokenExecutor):
                    pool_broken = True
                finally:
                    pbar.close()

                if pool_broken:
                    retry = [ww for ww in batch if ww[0] not in finished_in_batch]
                    pending = retry + pending
                    motion_workers = 1
                    log.warn(
                        f"WARNING: process pool crashed; retrying {len(retry)} motion(s) "
                        "sequentially"
                    )

    meta = {
        "hml_root": str(hml_root),
        "out_root": str(out_root),
        "b3d_subdir": NIMBLE_B3D_SUBDIR,
        "nimble_b3d_dir": str(b3d_cache),
        "fps": fps,
        "joint_source": str(getattr(args, "joint_source", "auto")),
        "joints_root": joints_root or None,
        "num_dofs": num_dofs_ref,
        "motions_ok": ok,
        "motions_error": err,
        "motions_skipped": skip,
        "activation_method": activation_method,
        "skip_muscle_activation": skip_muscle_activation,
        "muscle_activation_config": muscle_activation_config_to_dict(act_cfg),
        "moco_threads": moco_threads if activation_method == "moco_track" else None,
        "motion_process_workers": motion_workers,
        "moco_parallel_motions": (
            moco_parallel if activation_method == "moco_track" else None
        ),
        "num_shards": num_shards if num_shards > 1 else None,
        "shard_index": shard_index if num_shards > 1 else None,
        "isolate_motion_process": isolate_motion_process,
        "manifest_path": str(manifest_path),
        "use_all_frames_per_motion": True,
    }
    run_log_file = getattr(args, "_run_log_file", None)
    if run_log_file:
        meta["run_log_file"] = str(run_log_file)
    if num_shards <= 1:
        (out_root / "preprocess_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    log.progress(f"Done: {ok} ok, {err} failed, {skip} skipped")
    log.verbose(f"B3D cache: {b3d_cache}")
    log.verbose(f"Activation method: {activation_method}")
    log.verbose(f"Successful: {ok}")
    log.verbose(f"Unsuccessful: {err}")
    log.verbose(f"Skipped: {skip}")
    if loss_count > 0:
        log.verbose(f"Average IK Fit: {loss_sum / loss_count:.6f}")
    if moco_obj_count > 0:
        log.verbose(
            f"Average Muscle Activations: {moco_obj_sum / moco_obj_count:.6f}"
        )

    if ok == 0 and skip == 0:
        sys.exit(1)

    if not skip_normalization:
        compute_nimble_normalization_stats(out_root)
        removed = cleanup_preprocess_manifests(out_root)
        if removed:
            log.verbose(f"Removed {len(removed)} temporary preprocess manifest file(s)")

    from nimble.export import clear_export_caches

    clear_export_caches()


def main() -> None:
    parser = argparse.ArgumentParser(description="HumanML3D → Nimble B3D preprocessing")
    default_root = default_humanml3d_root()
    parser.add_argument(
        "--hml_root",
        default=default_root,
        help="HumanML3D root (new_joints or new_joint_vecs; default datasets/HumanML3D)",
    )
    parser.add_argument(
        "--out_root",
        default=default_root,
        help=f"Dataset root; B3D motions go to <out_root>/{NIMBLE_B3D_SUBDIR}/",
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--mass_kg", type=float, default=70.0)
    parser.add_argument("--height_m", type=float, default=1.75)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "With Moco (default): total CPU budget for Moco/Ipopt threads (0 = auto). "
            "Split across --moco_parallel_motions when >1. "
            "With --skip_muscle_activation: parallel IK-only workers (0 = auto)."
        ),
    )
    parser.add_argument(
        "--moco_parallel_motions",
        type=int,
        default=1,
        help=(
            "Run this many Moco solves concurrently (default 1). "
            "Threads per solve = num_workers // moco_parallel_motions."
        ),
    )
    parser.add_argument(
        "--skip_muscle_activation",
        action="store_true",
        help="Alias for --activation_method none (IK-only parallel workers).",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip motions that already have a .b3d in nimble_b3d/",
    )
    parser.add_argument(
        "--max_motions",
        type=int,
        default=0,
        help="Process at most this many motions (0 = all; useful for smoke tests)",
    )
    parser.add_argument(
        "--joint_source",
        choices=("auto", "joints", "new_joints"),
        default="auto",
        help="Joint XYZ input: auto prefers ./joints/ then new_joints/",
    )
    parser.add_argument(
        "--joints_root",
        default="",
        help="Directory with pre-normalization joints/ (default: <hml_root>/joints or HUMANML3D_JOINTS_ROOT)",
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
        "--skip_normalization",
        action="store_true",
        help="Skip Mean.npy/Std.npy (automatic when num_shards > 1; run compute_normalization.py after)",
    )
    add_muscle_activation_cli_args(parser)
    add_run_log_cli_args(parser)
    args = parser.parse_args()

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
    ) as (
        paths,
        logger,
    ):
        args._run_log_file = str(paths.log_file)
        logger.progress(f"log: {paths.latest_log}")
        run_preprocess(args, logger)


if __name__ == "__main__":
    main()
