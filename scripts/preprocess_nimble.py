#!/usr/bin/env python3
"""Preprocess HumanML3D motion into Nimble B3D cache (one .b3d per motion).

Run from the SINDyffuse repo root, e.g.:

  python scripts/preprocess_nimble.py
  python scripts/preprocess_nimble.py --num_workers 32   # Moco uses 32 threads (default: auto)
  python scripts/preprocess_nimble.py --skip_if_moco_valid   # resume: skip B3D with valid Moco mask
  python scripts/preprocess_nimble.py --motion_shard_index 0 --motion_shard_count 8
  python scripts/preprocess_nimble.py --moco_parallel_motions 4 --num_workers 64
  python scripts/preprocess_nimble.py --skip_muscle_activation --num_workers 8   # IK-only, 8 parallel
  python scripts/preprocess_nimble.py --max_motions 1   # smoke test

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

try:
    from concurrent.futures.process import BrokenProcessPool
except ImportError:
    BrokenProcessPool = BrokenExecutor  # type: ignore[misc, assignment]
from pathlib import Path

import numpy as np
from tqdm import tqdm

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
from common.cpu import configure_compute_threads, detect_usable_cpus, resolve_preprocess_parallelism
from common.paths import NIMBLE_B3D_SUBDIR, default_humanml3d_root, nimble_b3d_dir
from common.run_logging import (
    RunLogger,
    add_run_log_cli_args,
    append_verbose_log,
    null_logger,
    run_log_session,
)
from datasets.hml3d_joints import default_joints_root, load_hml3d_joint_positions
from datasets.nimble_dataset import compute_nimble_normalization_stats
from datasets.splits import all_motion_ids
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
    bool,
    float,
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


def _b3d_activation_valid_fraction(b3d_path: Path) -> float | None:
    """Mean muscle_activation_mask if present; ``None`` when activation labels are absent."""
    if not b3d_path.is_file():
        return None
    try:
        import nimblephysics as nimble

        from nimble.b3d_io import read_muscle_activation_mask_frames, subject_has_custom_value
        from nimble.b3d_schema import MUSCLE_ACTIVATION_MASK

        subj = nimble.biomechanics.SubjectOnDisk(str(b3d_path))
        if not subject_has_custom_value(subj, MUSCLE_ACTIVATION_MASK):
            return None
        tlen = int(subj.getTrialLength(0))
        if tlen < 1:
            return None
        mask = read_muscle_activation_mask_frames(subj, 0, 0, tlen)
        return float(np.mean(mask))
    except Exception:
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
        skip_if_moco_valid,
        moco_valid_fraction_threshold,
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
    if skip_if_moco_valid and out_b3d.is_file():
        valid_frac = _b3d_activation_valid_fraction(out_b3d)
        if valid_frac is not None and valid_frac >= float(moco_valid_fraction_threshold):
            append_verbose_log(
                f"{sid}: skipped (activation_valid fraction={valid_frac:.4f})"
            )
            return {
                "id": sid,
                "status": "skipped",
                "path": str(out_b3d),
                "skip_reason": "activation_valid",
                "activation_valid_fraction": valid_frac,
            }

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
    """File-only per-motion summary (IK loss, Moco objective, timing)."""
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
    label_frac = ik.get("muscle_activation_label_valid_fraction")
    if label_frac is not None:
        logger.verbose(f"{mid}: label_valid_fraction {float(label_frac):.4f}")
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
    """Terminal: IK / muscle-activation errors only (metrics go to the run log)."""
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
    label_frac = ik.get("muscle_activation_label_valid_fraction")
    if label_frac is not None and float(label_frac) < 0.5:
        logger.warn(f"{mid} muscle activation low valid fraction: {float(label_frac):.4f}")


def _motion_loss(row: dict) -> float | None:
    if row.get("status") != "ok":
        return None
    return _ik_fit_loss(row.get("ik_stats") or {})


def _migrate_legacy_b3d_moco(out_root: Path, logger: RunLogger) -> None:
    """Merge ``nimble_b3d_moco/`` into canonical ``nimble_b3d/`` (idempotent)."""
    legacy_dir = out_root / "nimble_b3d_moco"
    target_dir = nimble_b3d_dir(out_root)
    if not legacy_dir.is_dir():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    migrated = 0
    for src in sorted(legacy_dir.glob("*.b3d")):
        dst = target_dir / src.name
        if not dst.is_file():
            shutil.copy2(src, dst)
            migrated += 1
            continue
        src_frac = _b3d_activation_valid_fraction(src) or 0.0
        dst_frac = _b3d_activation_valid_fraction(dst) or 0.0
        if src_frac > dst_frac:
            shutil.copy2(src, dst)
            migrated += 1
    legacy_manifest = out_root / "preprocess_manifest_nimble_b3d_moco.jsonl"
    manifest_path = out_root / "preprocess_manifest.jsonl"
    if legacy_manifest.is_file():
        existing_ids: set[str] = set()
        if manifest_path.is_file():
            for line in manifest_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    existing_ids.add(str(json.loads(line).get("id", "")))
                except json.JSONDecodeError:
                    pass
        with manifest_path.open("a", encoding="utf-8") as out_fp:
            for line in legacy_manifest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row_id = str(json.loads(line).get("id", ""))
                except json.JSONDecodeError:
                    continue
                if row_id and row_id not in existing_ids:
                    out_fp.write(line + "\n")
                    existing_ids.add(row_id)
    if migrated:
        logger.verbose(f"Migrated {migrated} .b3d file(s) from nimble_b3d_moco/ to nimble_b3d/")


def _symlink_metadata(hml_root: Path, out_root: Path) -> None:
    if hml_root.resolve() == out_root.resolve():
        return
    src_texts = hml_root / "texts"
    if src_texts.is_dir():
        _link_or_copy_tree(src_texts, out_root / "texts")
    for name in ("train.txt", "val.txt", "test.txt"):
        src = hml_root / name
        dst = out_root / name
        if src.is_file() and not dst.exists():
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)


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
    _migrate_legacy_b3d_moco(out_root, log)

    ids = all_motion_ids(hml_root)
    max_motions = int(getattr(args, "max_motions", 0) or 0)
    if max_motions > 0:
        ids = ids[:max_motions]

    shard_count = int(getattr(args, "motion_shard_count", 1) or 1)
    shard_index = int(getattr(args, "motion_shard_index", 0) or 0)
    if shard_count > 1:
        if shard_index < 0 or shard_index >= shard_count:
            log.progress(
                f"ERROR: motion_shard_index={shard_index} must be in [0, {shard_count})"
            )
            sys.exit(1)
        ids = [mid for i, mid in enumerate(ids) if i % shard_count == shard_index]
        log.verbose(f"Motion shard {shard_index}/{shard_count}: {len(ids)} motion(s)")

    _symlink_metadata(hml_root, out_root)

    joints_root = str(getattr(args, "joints_root", "") or "").strip()
    if not joints_root:
        jr = default_joints_root(hml_root)
        joints_root = str(jr) if jr is not None else ""

    fps = float(getattr(args, "fps", 20.0))
    mass_kg = float(getattr(args, "mass_kg", 70.0))
    height_m = float(getattr(args, "height_m", 1.75))
    skip_existing = bool(getattr(args, "skip_existing", False))
    skip_if_activation_valid = bool(
        getattr(args, "skip_if_activation_valid", False)
        or getattr(args, "skip_if_moco_valid", False)
    )
    valid_threshold = getattr(args, "activation_valid_fraction_threshold", None)
    if valid_threshold is None:
        valid_threshold = getattr(args, "moco_valid_fraction_threshold", 0.5)
    moco_valid_fraction_threshold = float(valid_threshold)
    skip_if_moco_valid = skip_if_activation_valid

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
            skip_if_moco_valid,
            moco_valid_fraction_threshold,
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
    use_tqdm = bool(sys.stdout.isatty()) and total_motions > 0

    def _record(row: dict, *, pbar: tqdm | None = None) -> None:
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
    )
    if activation_method != "none":
        configure_compute_threads(moco_threads)
    if activation_method == "none":
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

    manifest_name = "preprocess_manifest"
    if shard_count > 1:
        manifest_name += f"_shard{shard_index}of{shard_count}"
    manifest_path = out_root / f"{manifest_name}.jsonl"
    pending = list(work)
    with manifest_path.open("w", encoding="utf-8") as mf:
        while pending:
            if motion_workers <= 1:
                batch = pending
                pending = []
                pbar = tqdm(
                    total=len(batch),
                    desc="preprocess",
                    unit="motion",
                    disable=not use_tqdm,
                )
                try:
                    for w in batch:
                        row = _process_one(w)
                        mf.write(json.dumps(row, default=str) + "\n")
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
                pbar = tqdm(
                    total=len(batch),
                    desc="preprocess",
                    unit="motion",
                    disable=not use_tqdm,
                )
                try:
                    with ProcessPoolExecutor(**pool_kwargs) as ex:
                        futs = {ex.submit(_process_one, w): w for w in batch}
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
        "motion_shard_index": shard_index if shard_count > 1 else None,
        "motion_shard_count": shard_count if shard_count > 1 else None,
        "skip_if_activation_valid": skip_if_activation_valid,
        "skip_if_moco_valid": skip_if_activation_valid,
        "activation_valid_fraction_threshold": (
            moco_valid_fraction_threshold if skip_if_activation_valid else None
        ),
        "manifest_path": str(manifest_path),
        "use_all_frames_per_motion": True,
    }
    run_log_file = getattr(args, "_run_log_file", None)
    if run_log_file:
        meta["run_log_file"] = str(run_log_file)
    meta_name = "preprocess_meta"
    if shard_count > 1:
        meta_name += f"_shard{shard_index}of{shard_count}"
    (out_root / f"{meta_name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

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

    compute_nimble_normalization_stats(out_root)

    from nimble.physics import clear_cache

    clear_cache()


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
        "--verbose_progress",
        action="store_true",
        help="Deprecated (no-op): per-motion details are always written to the run log.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip motions that already have a .b3d in nimble_b3d/",
    )
    parser.add_argument(
        "--skip_if_activation_valid",
        action="store_true",
        help=(
            "Skip motions whose existing .b3d has muscle_activation_mask mean "
            "above --activation_valid_fraction_threshold."
        ),
    )
    parser.add_argument(
        "--skip_if_moco_valid",
        action="store_true",
        help="Alias for --skip_if_activation_valid.",
    )
    parser.add_argument(
        "--activation_valid_fraction_threshold",
        type=float,
        default=None,
        help="Min mean muscle_activation_mask to skip (default 0.5).",
    )
    parser.add_argument(
        "--moco_valid_fraction_threshold",
        type=float,
        default=0.5,
        help="Alias for --activation_valid_fraction_threshold.",
    )
    parser.add_argument(
        "--motion_shard_index",
        type=int,
        default=0,
        help="Shard index for parallel jobs (0 .. motion_shard_count-1).",
    )
    parser.add_argument(
        "--motion_shard_count",
        type=int,
        default=1,
        help="Number of motion shards for parallel K8s jobs (default 1 = no sharding).",
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
    add_muscle_activation_cli_args(parser)
    add_run_log_cli_args(parser)
    args = parser.parse_args()

    if args.no_run_log:
        run_preprocess(args, null_logger())
        return

    script_name = Path(__file__).stem
    with run_log_session(args.log_dir, script_name=script_name, argv=sys.argv) as (
        paths,
        logger,
    ):
        args._run_log_file = str(paths.log_file)
        logger.progress(f"log: {paths.latest_log}")
        run_preprocess(args, logger)


if __name__ == "__main__":
    main()
