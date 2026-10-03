#!/usr/bin/env python3
"""Build and publish LaiUhlrich2022 OpenSimAD polynomial caches once."""
from __future__ import annotations
import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common.cpu import configure_compute_threads
from common.memory_diagnostics import DIAGNOSTICS_ENV, diagnostic_event, diagnostic_stage, memory_monitor

def main() -> None:
    parser = argparse.ArgumentParser(description='Build reusable LaiUhlrich2022 OpenSimAD polynomial caches')
    parser.add_argument('--force', action='store_true', help='Rebuild scratch artifacts before publishing')
    parser.add_argument('--mode', choices=('single', 'prepare', 'extract', 'finalize'), default='single')
    parser.add_argument('--build_dir', default='', help='Immutable input/results directory on shared storage (distributed modes)')
    parser.add_argument('--chunk_index', type=int, default=None, help='Slice index (default: JOB_COMPLETION_INDEX)')
    parser.add_argument('--expected_chunks', type=int, default=None, help='Validate prepared task count against Indexed Job completions')
    parser.add_argument('--allow_fitting_upgrade', action='store_true', help='Finalize only: permit the known 5D-to-6D fitting bugfix with original slices')
    parser.add_argument('--num_threads', type=int, default=1, help='MuscleAnalysis workers (default: 1; increase only after measuring peak RSS)')
    parser.add_argument('--work_dir', default='', help='Scratch build directory (default: TMPDIR)')
    parser.add_argument('--chunk_frames', type=int, default=100, help='Frames per disposable OpenSim process (default: 100)')
    parser.add_argument('--log_dir', default=str(_REPO / 'logs'), help='Persistent directory for per-attempt JSONL diagnostics')
    parser.add_argument('--memory_log_interval', type=float, default=10.0, help='Memory heartbeat interval in seconds')
    args = parser.parse_args()
    if args.allow_fitting_upgrade and args.mode != 'finalize':
        parser.error('--allow_fitting_upgrade is only valid for finalize')
    if args.chunk_frames <= 0 or args.num_threads <= 0:
        parser.error('--chunk_frames and --num_threads must be positive')
    if args.mode != 'single' and not args.build_dir:
        parser.error('distributed modes require --build_dir on persistent shared storage')
    if args.mode != 'single' and args.force:
        parser.error('--force is only supported for single mode; use a new --build_dir for changed inputs')
    if args.expected_chunks is not None and args.expected_chunks <= 0:
        parser.error('--expected_chunks must be positive')
    if args.mode == 'extract' and args.chunk_index is None:
        index = os.environ.get('JOB_COMPLETION_INDEX', '')
        if not index.isdigit():
            parser.error('extract requires --chunk_index or JOB_COMPLETION_INDEX')
        args.chunk_index = int(index)
    if not math.isfinite(args.memory_log_interval) or args.memory_log_interval <= 0:
        parser.error('--memory_log_interval must be positive')
    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:12]
    diagnostics = log_dir / f'build_lai_opensimad_polynomials_{attempt}.jsonl'
    os.environ[DIAGNOSTICS_ENV] = str(diagnostics)
    print(f'Persistent polynomial diagnostics: {diagnostics}', flush=True)
    diagnostic_event('run_start', argv=sys.argv, arguments=vars(args), python=sys.version)
    with memory_monitor(float(args.memory_log_interval)), diagnostic_stage('polynomial_build'):
        try:
            revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=_REPO, capture_output=True,
                                      text=True, timeout=5, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            revision = None
        diagnostic_event('code_revision', git_head=revision)
        _build(args)
    diagnostic_event('run_complete')

def _build(args: argparse.Namespace) -> None:
    threads = max(1, int(args.num_threads))
    configure_compute_threads(threads)
    # Bind pip CasADi before polynomial_cache imports model_prep/OpenSim.
    import casadi  # noqa: F401
    import numpy as np
    import opensim
    diagnostic_event('runtime_versions', casadi=casadi.__version__, numpy=np.__version__,
                     opensim=str(getattr(opensim, '__version__', 'unknown')), configured_threads=threads)
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix='sindyffuse_polynomial_', dir=os.environ.get('TMPDIR')))
        cleanup = True
    try:
        if args.mode != 'single':
            from nimble.opensimad.polynomial_shards import prepare_shards, extract_shard, finalize_shards
            with diagnostic_stage(args.mode, build_dir=args.build_dir):
                if args.mode == 'prepare':
                    result = prepare_shards(Path(args.build_dir), chunk_frames=args.chunk_frames, expected_chunks=args.expected_chunks)
                elif args.mode == 'extract':
                    result = {'chunk_path': str(extract_shard(Path(args.build_dir), index=args.chunk_index,
                                work_dir=work_dir, expected_chunks=args.expected_chunks))}
                else:
                    result = finalize_shards(Path(args.build_dir), work_dir=work_dir, expected_chunks=args.expected_chunks,
                                             allow_fitting_upgrade=args.allow_fitting_upgrade)
            print(result, flush=True)
            diagnostic_event('distributed_mode_complete', mode=args.mode, result=result)
            return
        from nimble.opensimad.polynomial_cache import build_polynomial_cache
        result = build_polynomial_cache(
            work_dir=work_dir,
            num_threads=threads,
            chunk_frames=int(args.chunk_frames),
            force=bool(args.force),
        )
        print(f"Polynomial cache metadata: {result['metadata_path']}")
        print(f"MuscleAnalysis workers: {result['num_threads']}")
        print(f"MuscleAnalysis chunk frames: {result['chunk_frames']}")
        for name in result['artifacts']:
            print(f'  {name}')
        diagnostic_event('published_artifacts', result=result)
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)

if __name__ == '__main__':
    main()