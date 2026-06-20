#!/usr/bin/env python3
"""Quick benchmark: sequential vs parallel MocoTrack on short motion clips.

Uses truncated HumanML3D clips and relaxed Moco settings so each solve finishes in
minutes, not hours. Compares wall-clock time to process the same motion batch under
different ``moco_parallel_motions`` values (same total ``--num_workers`` budget).

Example (from repo root, sindyffuse env active):

  python scripts/benchmark_moco_parallel.py
  python scripts/benchmark_moco_parallel.py --num_workers 64 --parallel 1 2 4 8 16 32 64
  python scripts/benchmark_moco_parallel.py --num_motions 64 --num_frames 25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common.cpu import configure_compute_threads, detect_usable_cpus, resolve_preprocess_parallelism
from common.paths import default_humanml3d_root, nimble_b3d_dir
from common.run_logging import (
    RunLogger,
    add_run_log_cli_args,
    null_logger,
    run_log_session,
)
from datasets.hml3d_joints import default_joints_root, load_hml3d_joint_positions
from datasets.nimble_dataset import read_q_segment
from datasets.splits import all_motion_ids
from nimble.ik import fit_q
from nimble.physics import load_model
from nimble.skeleton_registry import get_spec
from nimble.muscle_activation import (
    MuscleActivationConfig,
    compute_muscle_activation,
    configure_opensim_logging,
    muscle_activation_config_from_dict,
    muscle_activation_config_to_dict,
)


def _fast_benchmark_cfg(base: MuscleActivationConfig) -> MuscleActivationConfig:
    """Relaxed Moco settings for timing only (not production quality)."""
    return replace(
        base,
        activation_method="moco_track",
        mesh_interval=0.1,
        moco_adaptive_mesh=False,
        moco_max_iterations=100,
        moco_convergence_tolerance=0.02,
        moco_min_frames=10,
        opensim_log_level="Off",
    )


def _load_clips(
    hml_root: Path,
    motion_ids: list[str],
    *,
    num_frames: int,
    joint_source: str,
    joints_root: Path | None,
    from_b3d: bool,
) -> list[tuple[str, np.ndarray]]:
    clips: list[tuple[str, np.ndarray]] = []
    b3d_dir = nimble_b3d_dir(hml_root)
    sk = None
    spec = None
    for sid in motion_ids:
        q: np.ndarray | None = None
        b3d_path = b3d_dir / f"{sid}.b3d"
        if from_b3d and b3d_path.is_file():
            full = read_q_segment(str(b3d_path))
            tlen = int(full.shape[0])
            n = min(int(num_frames), tlen)
            # Middle segment tends to be more stable than the first frames.
            start = max(0, (tlen - n) // 2)
            q = np.ascontiguousarray(full[start : start + n], dtype=np.float64)
        if q is None:
            if sk is None:
                sk = load_model().skeleton
                spec = get_spec("rajagopal")
            joints, _ = load_hml3d_joint_positions(
                hml_root,
                sid,
                joint_source=joint_source,  # type: ignore[arg-type]
                joints_root=joints_root,
            )
            t = min(int(num_frames), int(joints.shape[0]))
            if t < 10:
                raise ValueError(f"{sid}: only {t} frames (need >= 10)")
            joints = joints[:t]
            poses_q, ik_stats = fit_q(joints, sk, ik_mapping=spec.ik_mapping)
            if int(ik_stats.get("success_count", 0)) < 2:
                raise RuntimeError(f"{sid}: IK failed ({ik_stats})")
            q = np.ascontiguousarray(poses_q.T, dtype=np.float64)
        clips.append((sid, q))
    return clips


def _moco_worker_init(log_level: str, moco_threads: int) -> None:
    configure_compute_threads(int(moco_threads))
    configure_opensim_logging(log_level)


def _run_one_moco(item: tuple[str, bytes, str, list[int]]) -> dict:
    sid, q_bytes, cfg_json, shape = item
    cfg = muscle_activation_config_from_dict(json.loads(cfg_json))
    q = np.frombuffer(q_bytes, dtype=np.float64).reshape(shape)
    t0 = time.perf_counter()
    try:
        result = compute_muscle_activation(q, cfg=cfg)
        sec = time.perf_counter() - t0
        return {
            "id": sid,
            "status": "ok",
            "seconds": sec,
            "repaired_frame_count": int(result.metadata.get("repaired_frame_count", 0)),
            "solver_success": bool(result.metadata.get("moco_solver_success", False)),
        }
    except Exception as exc:
        return {
            "id": sid,
            "status": "error",
            "seconds": time.perf_counter() - t0,
            "error": str(exc),
        }


def _run_config(
    clips: list[tuple[str, np.ndarray]],
    cfg: MuscleActivationConfig,
    *,
    num_workers: int,
    parallel: int,
) -> dict:
    motion_workers, moco_threads = resolve_preprocess_parallelism(
        num_workers,
        activation_method="moco_track",
        moco_parallel_motions=parallel,
    )
    cfg_json = json.dumps(muscle_activation_config_to_dict(cfg))
    work: list[tuple[str, bytes, str, list[int]]] = []
    for sid, q in clips:
        work.append((sid, q.tobytes(), cfg_json, list(q.shape)))

    t0 = time.perf_counter()
    rows: list[dict] = []
    if motion_workers <= 1:
        _moco_worker_init(cfg.opensim_log_level, moco_threads)
        for item in work:
            rows.append(_run_one_moco(item))
    else:
        with ProcessPoolExecutor(
            max_workers=motion_workers,
            initializer=_moco_worker_init,
            initargs=(cfg.opensim_log_level, moco_threads),
        ) as ex:
            futs = [ex.submit(_run_one_moco, item) for item in work]
            for fut in as_completed(futs):
                rows.append(fut.result())
    wall = time.perf_counter() - t0
    ok = sum(1 for r in rows if r.get("status") == "ok")
    return {
        "parallel": parallel,
        "motion_workers": motion_workers,
        "moco_threads": moco_threads,
        "wall_seconds": wall,
        "motions_ok": ok,
        "motions_error": len(rows) - ok,
        "per_motion_seconds": rows,
    }


def run_benchmark(args: argparse.Namespace, logger: RunLogger | None = None) -> dict:
    log = logger or null_logger()
    if int(np.__version__.split(".", maxsplit=1)[0]) >= 2:
        log.progress("ERROR: nimblephysics requires numpy<2")
        sys.exit(1)

    hml_root = Path(args.hml_root).expanduser().resolve()
    joints_root = default_joints_root(hml_root)
    ids = all_motion_ids(hml_root)[: int(args.num_motions)]
    if len(ids) < int(args.num_motions):
        log.progress(f"ERROR: need {args.num_motions} motions, found {len(ids)}")
        sys.exit(1)

    total_workers = int(args.num_workers) if int(args.num_workers) > 0 else detect_usable_cpus()
    cfg = _fast_benchmark_cfg(MuscleActivationConfig(fps=20.0))

    log.progress(f"Loading {len(ids)} clips x {args.num_frames} frames...")
    clips = _load_clips(
        hml_root,
        ids,
        num_frames=int(args.num_frames),
        joint_source=str(args.joint_source),
        joints_root=joints_root,
        from_b3d=bool(args.from_b3d),
    )
    log.progress(
        f"usable_cpus={detect_usable_cpus()} num_workers={total_workers} "
        f"motions={len(clips)} frames={args.num_frames}"
    )

    if args.warmup:
        log.progress("Warmup solve...")
        _moco_worker_init(cfg.opensim_log_level, total_workers)
        compute_muscle_activation(clips[0][1], cfg=cfg)

    results: list[dict] = []
    for parallel in args.parallel:
        p = max(1, int(parallel))
        motion_workers, threads = resolve_preprocess_parallelism(
            total_workers, activation_method="moco_track", moco_parallel_motions=p
        )
        effective_workers = min(motion_workers, len(clips))
        log.progress(
            f"moco_parallel_motions={p} ({effective_workers} motion(s) x {threads} thread(s) each)"
        )
        out = _run_config(clips, cfg, num_workers=total_workers, parallel=p)
        results.append(out)
        for row in sorted(out["per_motion_seconds"], key=lambda r: r["id"]):
            st = row.get("status")
            sec = row.get("seconds", 0.0)
            if st == "ok":
                log.progress(
                    f"  {row['id']}: {sec:.1f}s ok "
                    f"repaired={row.get('repaired_frame_count', 0)} "
                    f"solver_ok={row.get('solver_success')}"
                )
            else:
                log.progress(f"  {row['id']}: {sec:.1f}s error {row.get('error', '')[:120]}")
        log.progress(
            f"  wall={out['wall_seconds']:.1f}s ok={out['motions_ok']} err={out['motions_error']}"
        )

    best = min(results, key=lambda r: r["wall_seconds"])
    log.progress("=== summary ===")
    log.progress(f"{'parallel':>8} {'workers':>8} {'threads':>8} {'wall_s':>8} {'ok':>4}")
    for r in results:
        mark = " <-- fastest" if r is best else ""
        log.progress(
            f"{r['parallel']:8d} {r['motion_workers']:8d} {r['moco_threads']:8d} "
            f"{r['wall_seconds']:8.1f} {r['motions_ok']:4d}{mark}"
        )
    comparisons: list[dict] = []
    if len(results) > 1:
        seq = next(r for r in results if r["parallel"] == 1)
        for r in results:
            if r["parallel"] == 1:
                continue
            speedup = seq["wall_seconds"] / max(r["wall_seconds"], 1e-6)
            verdict = (
                "faster" if speedup > 1.05 else "slower" if speedup < 0.95 else "~same"
            )
            log.progress(f"parallel={r['parallel']} vs sequential: {speedup:.2f}x ({verdict})")
            comparisons.append(
                {
                    "parallel": r["parallel"],
                    "speedup_vs_parallel_1": speedup,
                    "verdict": verdict,
                }
            )

    payload = {
        "hml_root": str(hml_root),
        "num_workers": total_workers,
        "num_motions": int(args.num_motions),
        "num_frames": int(args.num_frames),
        "from_b3d": bool(args.from_b3d),
        "motion_ids": ids,
        "muscle_activation_config": muscle_activation_config_to_dict(cfg),
        "results": results,
        "comparisons": comparisons,
        "fastest_parallel": int(best["parallel"]),
    }
    out_json = Path(getattr(args, "out_json", "") or "").expanduser()
    if not str(out_json):
        out_json = Path(args.log_dir).expanduser().resolve() / "benchmark_moco_parallel_latest.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.progress(f"Wrote results: {out_json}")

    if any(r["motions_ok"] == 0 for r in results):
        sys.exit(1)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark moco_parallel_motions on short clips")
    parser.add_argument("--hml_root", default=default_humanml3d_root())
    parser.add_argument("--num_workers", type=int, default=0, help="Total CPU budget (0 = auto)")
    parser.add_argument(
        "--parallel",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
        help="moco_parallel_motions values to compare (use num_motions >= max for full sweep)",
    )
    parser.add_argument(
        "--num_motions",
        type=int,
        default=64,
        help="Short clips per config (set >= max --parallel to saturate all levels)",
    )
    parser.add_argument("--num_frames", type=int, default=25, help="Truncate each clip to this many frames")
    parser.add_argument(
        "--from_b3d",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load q from nimble_b3d/*.b3d when available (default: true)",
    )
    parser.add_argument("--warmup", action="store_true", help="Run one warmup solve before timing")
    parser.add_argument("--joint_source", choices=("auto", "joints", "new_joints"), default="auto")
    parser.add_argument(
        "--out_json",
        default="",
        help="JSON results path (default: <log_dir>/benchmark_moco_parallel_latest.json)",
    )
    add_run_log_cli_args(parser)
    args = parser.parse_args()

    if args.no_run_log:
        run_benchmark(args, null_logger())
        return

    script_name = Path(__file__).stem
    with run_log_session(args.log_dir, script_name=script_name, argv=sys.argv) as (
        paths,
        logger,
    ):
        logger.progress(f"log: {paths.latest_log}")
        run_benchmark(args, logger)


if __name__ == "__main__":
    main()
