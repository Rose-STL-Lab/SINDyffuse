"""Default filesystem locations (bundled model + optional BiomechAI dataset root)."""

from __future__ import annotations

import os
from pathlib import Path

_OSIM_PKG = Path(__file__).resolve().parent
# Shipped next to this package so the pipeline works from any clone path.
DEFAULT_MODEL_PATH: str = str(_OSIM_PKG / "FullBodyModel-4.0" / "Rajagopal2015.osim")


def _biomechai_root() -> Path:
    return Path(os.environ.get("BIOMECHAI_ROOT", "/mnt/BiomechAI"))


DEFAULT_HML3D_JOINTS_DIR: str = str(_biomechai_root() / "datasets" / "HumanML3D" / "new_joints")
DEFAULT_NPZ_EXPORT_DIR: str = str(_biomechai_root() / "datasets" / "HML3D_biomechanics_npz")
