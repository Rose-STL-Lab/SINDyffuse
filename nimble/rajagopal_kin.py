"""Keypoint index constants (nimblephysics-free)."""
from __future__ import annotations
from typing import Any, Tuple

FOOT_BODY_NAMES: Tuple[str, ...] = ('calcn_l', 'calcn_r')
KEYPOINT_JOINT_NAMES: Tuple[str, ...] = ('ground_pelvis', 'hip_l', 'hip_r', 'back', 'mtp_l', 'mtp_r')
NUM_KEYPOINTS = len(KEYPOINT_JOINT_NAMES)
IDX_PELVIS = 0
IDX_HIP_L = 1
IDX_HIP_R = 2
IDX_BACK = 3
IDX_FOOT_L = 4
IDX_FOOT_R = 5
COM_KEYPOINT_INDICES: Tuple[int, ...] = (IDX_PELVIS, IDX_HIP_L, IDX_HIP_R, IDX_BACK)


def clear_rajagopal_kin_cache() -> None:
    return None


def foot_body_indices(sk: Any | None = None) -> Tuple[int, int]:
    del sk
    raise RuntimeError('nimblephysics skeleton FK removed; use nimble.opensim_ik.body_world_positions')
