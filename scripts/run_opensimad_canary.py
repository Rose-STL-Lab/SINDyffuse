#!/usr/bin/env python3
"""Solve one OpenSimAD window before launching the indexed worker fleet."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timezone
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
from nimble.lai_coord_map import LAI_CACHE_DOF_NAMES
from nimble.opensimad.canary_selection import filtered_bounds_violations
from common.memory_diagnostics import DIAGNOSTICS_ENV, diagnostic_event, diagnostic_stage, memory_monitor

def _memory_value(name: str) -> int | None:
    path = Path('/sys/fs/cgroup') / name
    try:
        raw = path.read_text(encoding='utf-8').strip()
        return None if raw == 'max' else int(raw)
    except (OSError, ValueError):
        return None

def _select_motion(cache: Path, motion_id: str, min_frames: int, *, fps: float=20.0,
                   mesh_interval: float=0.02, max_candidates: int=128) -> tuple[Path, dict]:
    candidates = [cache / f'{motion_id}.npz'] if motion_id else sorted(cache.glob('*.npz'))[:max_candidates]
    best: tuple[float, Path, dict] | None = None
    for path in candidates:
        try:
            data = read_motion_npz(path, mmap=False)
        except Exception:
            continue
        q = np.asarray(data['q'])
        if q.ndim == 2 and q.shape[1] == len(LAI_CACHE_DOF_NAMES) and q.shape[0] >= min_frames and np.isfinite(q[:min_frames]).all():
            violations = filtered_bounds_violations(q[:min_frames], LAI_CACHE_DOF_NAMES,
                fps=float(data.get('fps', fps)), mesh_interval=mesh_interval)
            if violations:
                diagnostic_event('canary_candidate_rejected', motion_path=str(path), bounds_violations=violations)
                if motion_id:
                    raise ValueError(f'Explicit canary motion {motion_id} exceeds filtered polynomial bounds: {json.dumps(violations)}')
                continue
            score = float(np.std(q[:min_frames], axis=0).sum())
            if best is None or score > best[0]:
                best = (score, path, data)
    if best is not None:
        _, path, data = best
        print(f'Canary input selected: {path} (filtered polynomial bounds verified)', flush=True)
        diagnostic_event('canary_candidate_selected', motion_path=str(path), score=best[0])
        return path, data
    label = motion_id or f'any motion with at least {min_frames} frames'
    raise FileNotFoundError(f'No finite, in-domain canary NPZ for {label} under {cache}; scanned {len(candidates)} candidates')

def main() -> None:
    parser = argparse.ArgumentParser(description='Run one MinT-sized OpenSimAD canary solve')
    parser.add_argument('--data_root', default=default_humanml3d_root())
    parser.add_argument('--motion_id', default=os.environ.get('OPENSIMAD_CANARY_MOTION_ID', ''))
    parser.add_argument('--fps', type=float, default=20.0)
    parser.add_argument('--mesh_interval', type=float, default=0.02)
    parser.add_argument('--max_iterations', type=int, default=2500)
    parser.add_argument('--work_dir', default='')
    parser.add_argument('--max_candidates', type=int, default=128, help='Maximum sorted motions scanned for an in-domain diagnostic window')
    parser.add_argument('--log_dir', default=str(_REPO / 'logs'))
    args = parser.parse_args()
    if args.max_candidates <= 0:
        parser.error('--max_candidates must be positive')
    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:12]
    path = log_dir / f'opensimad_canary_{name}.jsonl'
    os.environ[DIAGNOSTICS_ENV] = str(path)
    print(f'Persistent canary diagnostics: {path}', flush=True)
    diagnostic_event('canary_run_start', arguments=vars(args))
    with memory_monitor(10), diagnostic_stage('canary'):
        _run(args)
    diagnostic_event('canary_run_complete')

def _run(args: argparse.Namespace) -> None:
    threads = configure_compute_threads(_BOOTSTRAP_MOCO_THREADS)
    artifacts = validate_opensimad_worker_artifacts(load_library=True, deep=True)
    core, buffer = segment_frame_counts(float(args.fps), core_s=1.4, buffer_s=0.14)
    solve_frames = core + 2 * buffer
    cache = lai_cache_dir(Path(args.data_root))
    path, data = _select_motion(cache, str(args.motion_id).strip(), solve_frames,
                              fps=float(args.fps), mesh_interval=float(args.mesh_interval), max_candidates=args.max_candidates)
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
        diagnostic_event('canary_report', report=report)
        if not ok or not np.isfinite(activations).any():
            raise RuntimeError(f'OpenSimAD canary failed; see {report_path}')
    finally:
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)

if __name__ == '__main__':
    main()