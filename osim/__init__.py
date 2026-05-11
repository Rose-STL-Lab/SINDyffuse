"""SINDyffuse biomechanics package (local name ``osim`` so ``import opensim`` resolves to OpenSim SDK).

Modules
-------
``pipeline`` / ``npz_io`` / ``cli`` / ``hml3d_export``
    HumanML3D → kinematic features + MocoTrack-derived columns → NPZ.
``features``
    Kinematic feature bank (Torch core; NumPy when poses are ndarray); contact proxies, smoothing metadata.
``moco_runtime`` / ``rajagopal_markers``
    Moco marker tracking against Rajagopal2015.
``guidance``
    Training-time OSIM guidance (torch loss + optional numpy Moco oracle stats).
"""

from __future__ import annotations

from osim.features import compute_features_from_hml3d_torch
from osim.npz_io import load_npz, npz_info, save_npz
from osim.pipeline import from_array, from_hml3d, from_hml3d_dir, from_poses

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
