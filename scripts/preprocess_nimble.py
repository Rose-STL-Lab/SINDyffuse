from __future__ import annotations
import argparse
import importlib.util
import sys
import warnings
from pathlib import Path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
warnings.filterwarnings('ignore', message='A NumPy version.*SciPy', category=UserWarning)
from common.preprocess_runner import add_common_preprocess_args
from common.run_setup import apply_preprocess_job_env
from common.run_logging import add_run_log_cli_args, null_logger, run_log_session
from nimble.muscle_activation import add_muscle_activation_cli_args, resolve_activation_method

def _import_script(stem: str):
    path = _REPO / 'scripts' / f'{stem}.py'
    spec = importlib.util.spec_from_file_location(stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main() -> None:
    parser = argparse.ArgumentParser(description='Deprecated: use preprocess_ik.py and preprocess_moco.py')
    add_common_preprocess_args(parser)
    parser.add_argument('--moco_parallel_motions', type=int, default=1)
    parser.add_argument('--skip_muscle_activation', action='store_true')
    add_muscle_activation_cli_args(parser)
    add_run_log_cli_args(parser)
    args = parser.parse_args()
    apply_preprocess_job_env(args)
    method = resolve_activation_method(args)
    print('WARNING: scripts/preprocess_nimble.py is deprecated. Use preprocess_ik.py (Job 1) and preprocess_moco.py (Job 3).', file=sys.stderr)
    preprocess_ik = _import_script('preprocess_ik')
    preprocess_moco = _import_script('preprocess_moco')

    def runner(a, logger):
        preprocess_ik.run_preprocess_ik(a, logger)
        if method != 'none' and not a.skip_muscle_activation:
            preprocess_moco.run_preprocess_moco(a, logger)
    if args.no_run_log:
        runner(args, null_logger())
        return
    with run_log_session(args.log_dir, script_name=Path(__file__).stem, argv=sys.argv) as (paths, logger):
        args._run_log_file = str(paths.log_file)
        runner(args, logger)

if __name__ == '__main__':
    main()
