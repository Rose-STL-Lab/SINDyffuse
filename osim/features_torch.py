"""Torch HumanML3D -> differentiable kinematic feature tensors."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from osim.contact_torch import estimate_contact_from_hml3d
from osim.kinematics_torch import fft_energy_ratio, finite_diff, safe_norm


def compute_features_from_hml3d_torch(
    poses: torch.Tensor,
    dt: float,
    include_poses: bool = False,
    *,
    fps: int = 20,
    sampling_frequency: float | None = None,
    smooth_poses: bool = False,
    smooth_cutoff_hz: float = 6.0,
    smooth_butterworth_order: int = 2,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if poses.ndim == 2 and poses.shape == (22, 3):
        poses = poses.unsqueeze(0)
    if poses.ndim != 3 or tuple(poses.shape[1:]) != (22, 3):
        raise ValueError(f"Expected poses shape [T,22,3], got {tuple(poses.shape)}")

    # Torch-only v1: smoothing intentionally omitted to stay deterministic differentiable.
    smooth_meta: Dict[str, Any] = {
        "enabled": bool(False),
        "method": "none",
        "cutoff_hz_requested": float(smooth_cutoff_hz),
        "cutoff_hz_effective": None,
        "sample_rate_hz": float(sampling_frequency if sampling_frequency is not None else fps),
        "butterworth_order": int(smooth_butterworth_order),
        "note": "smoothing deferred in torch path",
    }
    t = int(poses.shape[0])
    flat = poses.reshape(t, -1)
    velocities = finite_diff(flat, dt)
    accelerations = finite_diff(velocities, dt)
    torques = finite_diff(accelerations, dt)
    torque_rate = finite_diff(torques, dt)

    com_pos = poses[:, [0, 1, 2, 3, 6, 9], :].mean(dim=1)
    com_vel = finite_diff(com_pos, dt)
    com_acc = finite_diff(com_vel, dt)
    contact_feats = estimate_contact_from_hml3d(poses, com_acc, dt=dt)

    torque_power = (torques * velocities).sum(dim=1, keepdim=True)
    power_pos = torch.clamp(torque_power, min=0.0)
    power_neg = torch.clamp(torque_power, max=0.0)

    feats: Dict[str, torch.Tensor] = {
        "velocities": velocities,
        "accelerations": accelerations,
        "torques": torques,
        "torque_rate": torque_rate,
        "com_pos": com_pos,
        "com_vel": com_vel,
        "com_acc": com_acc,
        "contact_wrench": contact_feats["contact_wrench"],
        "contact_wrench_l2": contact_feats["contact_wrench_l2"],
        "contact_active": contact_feats["contact_active"],
        "inferred_contact_active_lr": contact_feats["inferred_contact_active_lr"],
        "torque_power": torque_power,
        "power_pos": power_pos,
        "power_neg": power_neg,
        "cum_pos_work": torch.cumsum(power_pos, dim=0) * float(dt),
        "cum_neg_work": torch.cumsum(power_neg, dim=0) * float(dt),
        "torque_l2": safe_norm(torques),
        "torque_rate_l2": safe_norm(torque_rate),
        "vel_l2": safe_norm(velocities),
        "acc_l2": safe_norm(accelerations),
        "jerk_l2": safe_norm(finite_diff(accelerations, dt)),
        "com_speed": safe_norm(com_vel, dim=1),
        "com_acc_l2": safe_norm(com_acc, dim=1),
        "com_jerk_l2": safe_norm(finite_diff(com_acc, dt), dim=1),
        "momentum_proxy_l2": safe_norm(velocities, dim=1),
        "ang_momentum_proxy_l2": safe_norm(flat * velocities, dim=1),
        "kinetic_proxy": 0.5 * (velocities * velocities).sum(dim=1, keepdim=True),
        "effort_proxy_l1": torch.abs(torques).sum(dim=1, keepdim=True),
        "velocity_spectral_ratio": fft_energy_ratio(velocities),
        "accel_spectral_ratio": fft_energy_ratio(accelerations),
        "torque_spectral_ratio": fft_energy_ratio(torques),
        "summary_mean_abs_tau": torch.mean(torch.abs(torques), dim=0, keepdim=True),
        "summary_std_tau": torch.std(torques, dim=0, keepdim=True, unbiased=False),
        "summary_mean_abs_vel": torch.mean(torch.abs(velocities), dim=0, keepdim=True),
        "summary_std_vel": torch.std(velocities, dim=0, keepdim=True, unbiased=False),
        "summary_contact_ratio": contact_feats["summary_contact_ratio"],
        "dt": torch.tensor([float(dt)], dtype=poses.dtype, device=poses.device),
    }
    if include_poses:
        feats["positions"] = flat
    return feats, smooth_meta

