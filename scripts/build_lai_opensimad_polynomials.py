#!/usr/bin/env python3
"""Build and publish LaiUhlrich2022 OpenSimAD polynomial caches once."""
from __future__ import annotations
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Bind pip CasADi before polynomial_cache imports model_prep/OpenSim.
import casadi  # noqa: F401
from common.cpu import configure_compute_threads, detect_usable_cpus

def main() -> None:
    parser = argparse.ArgumentParser(description='Build reusable LaiUhlrich2022 OpenSimAD polynomial caches')
    parser.add_argument('--force', action='store_true', help='Rebuild scratch artifacts before publishing')
    parser.add_argument('--num_threads', type=int, default=1, help='MuscleAnalysis workers (default: 1; increase only after measuring peak RSS)')
    parser.add_argument('--work_dir', default='', help='Scratch build directory (default: TMPDIR)')
    args = parser.parse_args()
    threads = max(1, int(args.num_threads))
    configure_compute_threads(threads)
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix='sindyffuse_polynomial_', dir=os.environ.get('TMPDIR')))
        cleanup = True
    try:
        from nimble.opensimad.polynomial_cache import build_polynomial_cache
        result = build_polynomial_cache(work_dir=work_dir, num_threads=threads, force=bool(args.force))
        print(f"Polynomial cache metadata: {result['metadata_path']}")
        print(f"MuscleAnalysis workers: {result['num_threads']}")
        for name in result['artifacts']:
            print(f'  {name}')
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)

if __name__ == '__main__':
    main()