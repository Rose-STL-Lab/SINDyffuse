"""Central, clone-friendly path defaults for SINDyffuse.

Layout (default)::

    <repo>/datasets/HumanML3D/          # dataset root (texts, splits, raw motion)
        new_joints/ | new_joint_vecs/
        Mean.npy, Std.npy               # HumanML3D 263-D normalization (source only)
        nimble_b3d/                   # offline IK cache + q-space training artifacts
            {motion_id}.b3d
            Mean.npy, Std.npy           # Nimble q normalization for training
"""

from __future__ import annotations

import os
from pathlib import Path

# Subdirectory under the HumanML3D root for per-motion Nimble B3D exports.
NIMBLE_B3D_SUBDIR = "nimble_b3d"

__all__ = [
    "NIMBLE_B3D_SUBDIR",
    "default_datasets_dir",
    "default_humanml3d_root",
    "nimble_b3d_dir",
    "repo_root",
    "resolve_data_root",
    "resolve_repo_path",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_datasets_dir() -> Path:
    return repo_root() / "datasets"


def default_humanml3d_root() -> str:
    explicit = os.environ.get("HUMANML3D_ROOT", "").strip()
    if explicit:
        return explicit
    return str(default_datasets_dir() / "HumanML3D")


def nimble_b3d_dir(data_root: str | Path) -> Path:
    """Path to ``…/HumanML3D/nimble_b3d/`` (per-motion ``.b3d`` + q ``Mean.npy`` / ``Std.npy``)."""
    return Path(data_root).expanduser().resolve() / NIMBLE_B3D_SUBDIR


def resolve_data_root(path: str | None) -> str:
    if path is None or str(path).strip() == "":
        return default_humanml3d_root()
    return str(resolve_repo_path(path))


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a config path relative to the repo root when not absolute."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (repo_root() / p).resolve()
