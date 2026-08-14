from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from common.paths import activation_surrogate_latest_link, default_humanml3d_root, nimble_b3d_dir, repo_root, resolve_data_root, resolve_repo_path, results_dir, sindy_latest_link
PINNED_OUT_DIR_ENV = 'SINDYFFUSE_TRAIN_OUT_DIR'
__all__ = ['PINNED_OUT_DIR_ENV', 'default_config_path', 'new_run_dir', 'require_nimble_b3d', 'require_nimble_normalization', 'require_sindy_checkpoint', 'require_surrogate_checkpoint', 'resolve_run_dir', 'resolve_training_data_root', 'apply_preprocess_job_env']

def default_config_path(name: str) -> Path:
    return repo_root() / 'configs' / name

def new_run_dir(family: str, *, guidance: str | None=None) -> Path:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    if family == 'sindy':
        return results_dir() / 'sindy' / 'runs' / ts
    if family == 'activation_surrogate':
        return results_dir() / 'activation_surrogate' / 'runs' / ts
    if family == 'diffusion':
        mode = str(guidance or '').strip().lower()
        if mode not in {'none', 'sindy', 'nimble'}:
            raise ValueError(f'diffusion runs require guidance=none|sindy|nimble, got {guidance!r}')
        return results_dir() / 'diffusion' / mode / 'runs' / ts
    raise ValueError(f'unknown run family: {family!r}')

def resolve_run_dir(output: str | Path | None, *, family: str, guidance: str | None=None) -> Path:
    if output is not None and str(output).strip():
        path = Path(output).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()
    pinned = os.environ.get(PINNED_OUT_DIR_ENV, '').strip()
    if pinned:
        path = Path(pinned).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f'{PINNED_OUT_DIR_ENV}={pinned} is not a directory')
        return path
    path = new_run_dir(family, guidance=guidance)
    path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve()
    os.environ[PINNED_OUT_DIR_ENV] = str(resolved)
    return resolved

def resolve_training_data_root(data_root: str | Path | None) -> str:
    return resolve_data_root(str(data_root).strip() if data_root else None)

def require_nimble_b3d(data_root: str | Path) -> Path:
    cache = nimble_b3d_dir(data_root)
    if not cache.is_dir():
        raise FileNotFoundError(f'Nimble B3D cache required at {cache}. Run scripts/preprocess_ik.py and scripts/preprocess_moco.py first.')
    return cache

def require_nimble_normalization(data_root: str | Path) -> None:
    cache = require_nimble_b3d(data_root)
    mean_path = cache / 'Mean.npy'
    std_path = cache / 'Std.npy'
    if not mean_path.is_file() or not std_path.is_file():
        raise FileNotFoundError(f'Missing {mean_path} or {std_path}. Run scripts/compute_normalization.py first.')

def require_sindy_checkpoint() -> Path:
    ckpt = sindy_latest_link() / 'text_to_xi.pt'
    if not ckpt.is_file():
        raise FileNotFoundError(f'Missing {ckpt}. Run scripts/train_sindy.py first.')
    return ckpt

def require_surrogate_checkpoint() -> Path:
    latest = activation_surrogate_latest_link()
    for name in ('latest.pt', 'best.pt'):
        path = latest / name
        if path.is_file():
            return path
    raise FileNotFoundError(f'Missing activation surrogate checkpoint under {latest}. Run scripts/train_surrogate.py first.')

def resolve_repo_checkpoint(cfg_value: str) -> str:
    raw = str(cfg_value).strip()
    if not raw:
        return raw
    return str(resolve_repo_path(raw))

def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, '').strip()
    return int(raw) if raw.isdigit() else int(default)

def env_flag(name: str, default: bool=False) -> bool:
    raw = os.environ.get(name, '').strip().lower()
    if not raw:
        return bool(default)
    return raw in {'1', 'true', 'yes', 'on'}

def apply_preprocess_job_env(args) -> None:
    if os.environ.get('PREPROCESS_NUM_SHARDS'):
        args.num_shards = env_int('PREPROCESS_NUM_SHARDS', int(getattr(args, 'num_shards', 1)))
    if os.environ.get('MAX_MOTIONS'):
        args.max_motions = env_int('MAX_MOTIONS', 0)
    if 'SKIP_EXISTING' in os.environ:
        args.skip_existing = env_flag('SKIP_EXISTING', default=True)
    if int(getattr(args, 'num_shards', 1)) > 1:
        args.skip_normalization = True
    if os.environ.get('OPENSIM_LOG_LEVEL'):
        args.opensim_log_level = os.environ['OPENSIM_LOG_LEVEL'].strip()
    if os.environ.get('MOCO_CORE_DURATION_S'):
        args.moco_core_duration_s = float(os.environ['MOCO_CORE_DURATION_S'].strip())
    if os.environ.get('MOCO_BUFFER_DURATION_S'):
        args.moco_buffer_duration_s = float(os.environ['MOCO_BUFFER_DURATION_S'].strip())
    if os.environ.get('MOCO_STITCH_BLEND_S'):
        args.moco_stitch_blend_s = float(os.environ['MOCO_STITCH_BLEND_S'].strip())
    if os.environ.get('MOCO_MAX_ITERATIONS'):
        args.moco_max_iterations = env_int('MOCO_MAX_ITERATIONS', 2500)
    elif os.environ.get('OPENSIMAD_MAX_ITERATIONS'):
        args.moco_max_iterations = env_int('OPENSIMAD_MAX_ITERATIONS', 2500)
    if os.environ.get('MOCO_CONVERGENCE_TOLERANCE'):
        args.moco_convergence_tolerance = float(os.environ['MOCO_CONVERGENCE_TOLERANCE'].strip())
    if os.environ.get('MOCO_MESH_INTERVAL'):
        args.moco_mesh_interval = float(os.environ['MOCO_MESH_INTERVAL'].strip())
    elif os.environ.get('OPENSIMAD_MESH_INTERVAL'):
        args.moco_mesh_interval = float(os.environ['OPENSIMAD_MESH_INTERVAL'].strip())
    if os.environ.get('MOCO_PARALLEL_SEGMENTS'):
        args.moco_parallel_segments = env_int('MOCO_PARALLEL_SEGMENTS', 6)
    elif os.environ.get('OPENSIMAD_PARALLEL_SEGMENTS'):
        args.moco_parallel_segments = env_int('OPENSIMAD_PARALLEL_SEGMENTS', 6)
    if os.environ.get('ACTIVATION_METHOD'):
        args.activation_method = os.environ['ACTIVATION_METHOD'].strip()
    log_dir = str(getattr(args, 'log_dir', '') or '').strip()
    if not log_dir:
        args.log_dir = str(repo_root() / 'logs')