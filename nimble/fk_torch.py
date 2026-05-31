"""Backward-compatible exports for Rajagopal keypoint FK (not HumanML3D)."""

from __future__ import annotations

from typing import Tuple

import torch

from nimble.rajagopal_kin import (
    body_origins,
    foot_body_indices,
    keypoints_torch,
)

# Legacy name: returns [T, K, 3] Rajagopal keypoints, not [T, 22, 3] HML layout.
forward_kinematics = keypoints_torch

__all__ = ["forward_kinematics", "body_origins", "foot_body_indices", "keypoints_torch"]
