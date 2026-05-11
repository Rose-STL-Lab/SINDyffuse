"""HumanML3D → kinematic feature tensors (Torch core; NumPy in/out when poses are ndarray)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch

from osim.contact import estimate_contact_from_hml3d
from osim.kinematics import fft_energy_ratio, finite_diff, safe_norm
from osim.manifest import sha256
from osim.smoothing import apply_pose_smoothing_torch


def smooth_method_channel_id(method: str) -> int:
    return {
        "none": 0,
        "butterworth_sosfiltfilt": 1,
        "fir_firwin_conv1d": 2,
    }.get(str(method), 0)


def _to_numpy_feats(feats: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in feats.items():
        arr = v.detach().cpu().numpy()
        if k == "model_path_hash32":
            out[k] = arr.astype(np.uint32)
        else:
            out[k] = arr.astype(np.float32)
    return out


def _compute_features_torch(
    poses: torch.Tensor,
    dt: float,
    include_poses: bool,
    model_path: str | None,
    *,
    fps: int,
    sampling_frequency: float | None,
    smooth_poses: bool,
    smooth_cutoff_hz: float,
    smooth_butterworth_order: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if poses.ndim == 2 and tuple(poses.shape) == (22, 3):
        poses = poses.unsqueeze(0)
    if poses.ndim != 3 or tuple(poses.shape[1:]) != (22, 3):
        raise ValueError(f"Expected poses shape [T,22,3], got {tuple(poses.shape)}")

    poses, smooth_meta = apply_pose_smoothing_torch(
        poses,
        fps=int(fps),
        sampling_frequency=sampling_frequency,
        smooth_poses=bool(smooth_poses),
        smooth_cutoff_hz=float(smooth_cutoff_hz),
        smooth_butterworth_order=int(smooth_butterworth_order),
    )
    sid = smooth_method_channel_id(str(smooth_meta.get("method", "none")))
    cutoff_req = float(smooth_meta["cutoff_hz_requested"])
    cutoff_eff = smooth_meta["cutoff_hz_effective"]

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

    inferred_lr = contact_feats["inferred_contact_active_lr"]
    inferred_sum = inferred_lr.sum(dim=1, keepdim=True)
    inferred_active = (inferred_sum > 0).to(poses.dtype)

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
        "contact_inferred": torch.tensor([1.0], dtype=poses.dtype, device=poses.device),
        "contact_source_moco_estimated_proxy": torch.tensor([1.0], dtype=poses.dtype, device=poses.device),
        "inferred_contact_active_lr": inferred_lr,
        "inferred_contact_active": inferred_active,
        "inferred_contact_force_total": contact_feats["contact_wrench"][:, 3:6] + contact_feats["contact_wrench"][:, 9:12],
        "inferred_contact_wrench_lr": contact_feats["contact_wrench"],
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
        "pose_smooth_enabled": torch.tensor(
            [1.0 if smooth_meta["enabled"] else 0.0], dtype=poses.dtype, device=poses.device
        ),
        "pose_smooth_method_id": torch.tensor([float(sid)], dtype=poses.dtype, device=poses.device),
        "pose_smooth_cutoff_hz_requested": torch.tensor([cutoff_req], dtype=poses.dtype, device=poses.device),
        "pose_smooth_cutoff_hz_effective": torch.tensor(
            [float(cutoff_eff) if cutoff_eff is not None else 0.0], dtype=poses.dtype, device=poses.device
        ),
        "pose_smooth_butterworth_order": torch.tensor(
            [float(smooth_meta.get("butterworth_order") or 0)], dtype=poses.dtype, device=poses.device
        ),
    }
    if include_poses:
        feats["positions"] = flat

    if model_path is not None:
        mp = Path(model_path)
        if mp.exists():
            feats["model_path_hash32"] = torch.tensor(
                [int(sha256(mp)[:8], 16)], dtype=torch.uint32, device=poses.device
            )
        else:
            feats["model_path_hash32"] = torch.tensor([0], dtype=torch.uint32, device=poses.device)

    return feats, smooth_meta


def compute_features_from_hml3d(
    poses: Union[np.ndarray, torch.Tensor],
    dt: float,
    include_poses: bool,
    model_path: str | None,
    *,
    fps: int,
    sampling_frequency: float | None,
    smooth_poses: bool,
    smooth_cutoff_hz: float,
    smooth_butterworth_order: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return feature dict on the same array stack as ``poses`` (NumPy or Torch)."""
    is_numpy = isinstance(poses, np.ndarray)
    if is_numpy:
        poses_t = torch.as_tensor(poses, dtype=torch.float32)
    else:
        poses_t = poses if poses.dtype == torch.float32 else poses.float()

    feats_t, meta = _compute_features_torch(
        poses_t,
        dt=float(dt),
        include_poses=bool(include_poses),
        model_path=model_path,
        fps=int(fps),
        sampling_frequency=sampling_frequency,
        smooth_poses=bool(smooth_poses),
        smooth_cutoff_hz=float(smooth_cutoff_hz),
        smooth_butterworth_order=int(smooth_butterworth_order),
    )

    if is_numpy:
        return _to_numpy_feats(feats_t), meta
    return feats_t, meta


compute_features_from_hml3d_torch = compute_features_from_hml3d
