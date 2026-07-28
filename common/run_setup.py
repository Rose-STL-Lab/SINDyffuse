"""Shared run-directory and input validation for training / preprocess entry points."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from common.paths import (
    activation_surrogate_latest_link,
    default_humanml3d_root,
    mint_cache_dir,
    motion_cache_dir,
    repo_root,
    resolve_data_root,
    resolve_repo_path,
    results_dir,
    sindy_latest_link,
)

PINNED_OUT_DIR_ENV = "SINDYFFUSE_TRAIN_OUT_DIR"

__all__ = [
    "PINNED_OUT_DIR_ENV",
    "default_config_path",
    "env_flag",
    "env_int",
    "new_run_dir",
    "require_mint_cache",
    "require_mint_normalization",
    "require_motion_cache",
    "require_motion_normalization",
    "require_sindy_checkpoint",
    "require_surrogate_checkpoint",
    "resolve_run_dir",
    "resolve_training_data_root",
    "apply_preprocess_job_env",
]


def default_config_path(name: str) -> Path:
    return repo_root() / "configs" / name


def new_run_dir(family: str, *, guidance: str | None = None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if family == "sindy":
        return results_dir() / "sindy" / "runs" / ts
    if family == "activation_surrogate":
        return results_dir() / "activation_surrogate" / "runs" / ts
    if family == "diffusion":
        mode = str(guidance or "").strip().lower()
        if mode not in {"none", "sindy"}:
            raise ValueError(f"diffusion runs require guidance=none|sindy, got {guidance!r}")
        return results_dir() / "diffusion" / mode / "runs" / ts
    raise ValueError(f"unknown run family: {family!r}")


def resolve_run_dir(output: str | Path | None, *, family: str, guidance: str | None = None) -> Path:
    if output is not None and str(output).strip():
        path = Path(output).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    pinned = os.environ.get(PINNED_OUT_DIR_ENV, "").strip()
    if pinned:
        path = Path(pinned).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"{PINNED_OUT_DIR_ENV}={pinned} is not a directory")
        return path

    path = new_run_dir(family, guidance=guidance)
    path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve()
    os.environ[PINNED_OUT_DIR_ENV] = str(resolved)
    return resolved


def resolve_training_data_root(data_root: str | Path | None) -> str:
    return resolve_data_root(str(data_root).strip() if data_root else None)


def require_mint_cache(data_root: str | Path) -> Path:
    cache = mint_cache_dir(data_root)
    if not cache.is_dir():
        raise FileNotFoundError(
            f"MinT NPZ cache required at {cache}. Run scripts/preprocess_mint.py first."
        )
    return cache


def require_mint_normalization(data_root: str | Path) -> None:
    cache = require_mint_cache(data_root)
    mean_path = cache / "Mean.npy"
    std_path = cache / "Std.npy"
    if not mean_path.is_file() or not std_path.is_file():
        raise FileNotFoundError(
            f"Missing {mean_path} or {std_path}. "
            "Run scripts/compute_normalization.py."
        )


def require_motion_cache(data_root: str | Path) -> Path:
    return require_mint_cache(data_root)


def require_motion_normalization(data_root: str | Path) -> None:
    require_mint_normalization(data_root)


def require_sindy_checkpoint() -> Path:
    ckpt = sindy_latest_link() / "text_to_xi.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing {ckpt}. Run scripts/train_sindy.py first.")
    return ckpt


def require_surrogate_checkpoint() -> Path:
    latest = activation_surrogate_latest_link()
    for name in ("latest.pt", "best.pt"):
        path = latest / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Missing activation surrogate checkpoint under {latest}. Run scripts/train_surrogate.py first."
    )


def resolve_repo_checkpoint(cfg_value: str) -> str:
    raw = str(cfg_value).strip()
    if not raw:
        return raw
    return str(resolve_repo_path(raw))


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw.isdigit() else int(default)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def apply_preprocess_job_env(args) -> None:
    """Apply Kubernetes job env vars to preprocess CLI args."""
    if os.environ.get("PREPROCESS_NUM_SHARDS"):
        args.num_shards = env_int("PREPROCESS_NUM_SHARDS", int(getattr(args, "num_shards", 1)))
    if os.environ.get("MAX_MOTIONS"):
        args.max_motions = env_int("MAX_MOTIONS", 0)
    if "SKIP_EXISTING" in os.environ:
        args.skip_existing = env_flag("SKIP_EXISTING", default=True)
    if int(getattr(args, "num_shards", 1)) > 1:
        args.skip_normalization = True
    if os.environ.get("OPENSIM_LOG_LEVEL"):
        args.opensim_log_level = os.environ["OPENSIM_LOG_LEVEL"].strip()
    log_dir = str(getattr(args, "log_dir", "") or "").strip()
    if not log_dir:
        args.log_dir = str(repo_root() / "logs")
