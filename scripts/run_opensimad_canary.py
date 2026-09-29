#!/usr/bin/env python3
"""Solve one OpenSimAD window before launching the indexed worker fleet."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# CasADi must bind before OpenSim in spawned and parent processes.
import casadi  # noqa: F401
import numpy as np

from common.cpu import bootstrap_moco_compute_threads, configure_compute_threads
_BOOTSTRAP_MOCO_THREADS = bootstrap_moco_compute_threads(num_workers=1, num_shards=1)
from common.paths import default_humanml3d_root, lai_cache_dir
from datasets.lai_cache import read_motion_npz
from nimble.moco_segment import segment_frame_counts
from nimble.muscle_activation import MuscleActivationConfig, muscle_activation_config_to_dict
from nimble.opensim_ik import apply_ground_offset_lai_q
from nimble.opensimad.paths import opensimad_dir, validate_opensimad_worker_artifacts
from nimble.opensimad_track import solve_one_opensimad_segment_isolated

def _memory_value(name: str) -> int | None:
    path = Path('/sys/fs/cgroup') / name
    try:
        raw = path.read_text(encoding='utf-8').strip()
        return None if raw == 'max' else int(raw)
    except (OSError, ValueError):
        return None

def _select_motion(cache: Path, motion_id: str, min_frames: int) -> tuple[Path, dict]:
    candidates = [cache / f'{motion_id}.npz'] if motion_id else sorted(cache.glob('*.npz'))[:128]
    best: tuple[float, Path, dict] | None = None
    for path in candidates:
        try:
            data = read_motion_npz(path, mmap=False)
        except Exception:
            continue
        q = np.asarray(data['q'])
        if q.ndim == 2 and q.shape[0] >= min_frames and np.isfinite(q[:min_frames]).all():
            score = float(np.std(q[:min_frames], axis=0).sum())
            if best is None or score > best[0]:
                best = (score, path, data)
    if best is not None:
        _, path, data = best
        return path, data
    label = motion_id or f'any motion with at least {min_frames} frames'
    raise FileNotFoundError(f'No valid canary NPZ for {label} under {cache}')

def main() -> None:
    parser = argparse.ArgumentParser(description='Run one MinT-sized OpenSimAD canary solve')
    parser.add_argument('--data_root', default=default_humanml3d_root())
    parser.add_argument('--motion_id', default=os.environ.get('OPENSIMAD_CANARY_MOTION_ID', ''))
    parser.add_argument('--fps', type=float, default=20.0)
    parser.add_argument('--mesh_interval', type=float, default=0.02)
    parser.add_argument('--max_iterations', type=int, default=2500)
    parser.add_argument('--work_dir', default='')
    args = parser.parse_args()

    threads = configure_compute_threads(_BOOTSTRAP_MOCO_THREADS)
    artifacts = validate_opensimad_worker_artifacts(load_library=True, deep=True)
    core, buffer = segment_frame_counts(float(args.fps), core_s=1.4, buffer_s=0.14)
    solve_frames = core + 2 * buffer
    cache = lai_cache_dir(Path(args.data_root))
    path, data = _select_motion(cache, str(args.motion_id).strip(), solve_frames)
    fps = float(data.get('fps', args.fps))
    q = np.asarray(data['q'][:solve_frames], dtype=np.float64)
    q, ground_shift = apply_ground_offset_lai_q(q, sphere_offset_y_m=-0.02)

    if args.work_dir:
        work = Path(args.work_dir).expanduser().resolve()
        work.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work = Path(tempfile.mkdtemp(prefix='sindyffuse_canary_', dir=os.environ.get('TMPDIR')))
        cleanup = True
    try:
        cfg = MuscleActivationConfig(
            activation_method='opensimad', fps=fps,
            mass_kg=float(data.get('mass_kg', 70.0)),
            mesh_interval=float(args.mesh_interval),
            moco_max_iterations=int(args.max_iterations),
            moco_parallel_segments=1,
            opensim_log_level='Warn',
        )
        job = (0, q, muscle_activation_config_to_dict(cfg), str(work / 'segment_0000'), float(args.mesh_interval))
        _, activations, ok, metadata, grf = solve_one_opensimad_segment_isolated(job)
        report = {
            'motion_id': str(data.get('motion_id', path.stem)),
            'motion_path': str(path),
            'solve_frames': solve_frames,
            'fps': fps,
            'ground_offset_m': float(ground_shift),
            'success': bool(ok),
            'finite_activation_values': int(np.isfinite(activations).sum()),
            'finite_grf_values': int(np.isfinite(grf).sum()),
            'solver_metadata': metadata,
            'memory_current_bytes': _memory_value('memory.current'),
            'memory_peak_bytes': _memory_value('memory.peak'),
            'memory_limit_bytes': _memory_value('memory.max'),
            'configured_threads': threads,
            'external_library': artifacts['external_function']['library'],
        }
        report_path = opensimad_dir() / 'canary_report.json'
        report_path.write_text(json.dumps(report, indent=2, default=str) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2, default=str))
        if not ok or not np.isfinite(activations).any():
            raise RuntimeError(f'OpenSimAD canary failed; see {report_path}')
    finally:
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)

if __name__ == '__main__':
    main()