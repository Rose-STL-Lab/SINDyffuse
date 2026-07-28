"""Biomechanical target channel names shared across SINDy and guidance."""

from __future__ import annotations

from typing import Tuple

# All channels in L_bio (each is a per-frame magnitude / penalty scalar).
BIOMECH_COMPONENT_KEYS: Tuple[str, ...] = (
    # DOF (generalized coordinates)
    "vel",
    "acc",
    "torque",
    "torque_rate",
    "jerk",
    "effort",
    "joint_limit",
    "kinetic_q",
    "torque_power",
    # CoM (pose kinematics)
    "com_speed",
    "com_acc",
    "com_jerk",
    # Contact / GRF (pose + foot FK from q)
    "contact_gap",
    "contact_wrench",
    "grf_left",
    "grf_right",
    "grf_vertical",
    "grf_weight_deficit",
    "foot_slip",
    # Pose-space proxies
    "pose_vel",
    "pose_acc",
    "ang_momentum",
)

__all__ = ["BIOMECH_COMPONENT_KEYS"]
