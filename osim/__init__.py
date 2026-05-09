"""HumanML3D OpenSim-kinematic feature pipeline (package ``osim``).

The folder is named ``osim`` so ``import opensim`` continues to resolve to the OpenSim SDK.
"""

from __future__ import annotations

from osim.npz_io import load_npz, npz_info, save_npz
from osim.pipeline import from_array, from_hml3d, from_hml3d_dir, from_poses
from osim.features_torch import compute_features_from_hml3d_torch

__all__ = [
    "compute_features_from_hml3d_torch",
    "from_array",
    "from_hml3d",
    "from_hml3d_dir",
    "from_poses",
    "load_npz",
    "npz_info",
    "save_npz",
]
