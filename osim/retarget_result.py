"""Retargeting result container."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RetargetResult:
    q: np.ndarray  # [T, ndof]
    mean_fk_error: float
    num_frames: int
    method: str = "mint"
    mean_smpl_joint_error_m: float = 0.0
