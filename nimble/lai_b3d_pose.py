"""Deprecated: 37-DOF Rajagopal B3D pose embed removed with nimblephysics."""
from __future__ import annotations
from typing import Any
import numpy as np


def lai_q_to_b3d_pose(*_a: Any, **_k: Any) -> np.ndarray:
    raise RuntimeError('lai_b3d_pose removed; store Lai 31-DOF q in lai_cache NPZ')


def b3d_pose_to_lai_q(*_a: Any, **_k: Any) -> np.ndarray:
    raise RuntimeError('lai_b3d_pose removed; store Lai 31-DOF q in lai_cache NPZ')
