"""Central, clone-friendly path defaults for SINDyffuse.

All defaults are anchored to :func:`repo_root`, so clones work on any machine
without hard-coded absolute prefixes.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "DEFAULT_HML3D_JOINTS_DIR",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_NPZ_EXPORT_DIR",
    "checkpoints_dir",
    "default_datasets_dir",
    "default_humanml3d_root",
    "repo_root",
    "resolve_data_root",
    "runs_dir",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_datasets_dir() -> Path:
    return repo_root() / "datasets"


def checkpoints_dir() -> Path:
    p = os.environ.get("SINDYFFUSE_CHECKPOINTS_DIR", "").strip()
    return Path(p) if p else repo_root() / "checkpoints"


def runs_dir() -> Path:
    p = os.environ.get("SINDYFFUSE_RUNS_DIR", "").strip()
    return Path(p) if p else repo_root() / "runs"


def default_humanml3d_root() -> str:
    explicit = os.environ.get("HUMANML3D_ROOT", "").strip()
    if explicit:
        return explicit
    return str(default_datasets_dir() / "HumanML3D")


def resolve_data_root(path: str | None) -> str:
    if path is None or str(path).strip() == "":
        return default_humanml3d_root()
    p = Path(path)
    if p.is_absolute():
        return str(p.resolve())
    return str((repo_root() / p).resolve())


DEFAULT_MODEL_PATH: str = str(
    repo_root() / "osim" / "FullBodyModel-4.0" / "Rajagopal2015.osim"
)
DEFAULT_HML3D_JOINTS_DIR: str = str(Path(default_humanml3d_root()) / "new_joints")
DEFAULT_NPZ_EXPORT_DIR: str = str(default_datasets_dir() / "HML3D_biomechanics_npz")
