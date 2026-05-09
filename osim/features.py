"""HumanML3D → kinematic feature tensors (OpenSim-adjacent provenance)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from osim.contact import estimate_contact_from_hml3d
from osim.kinematics import fft_energy_ratio, finite_diff, safe_norm
from osim.manifest import sha256
from osim.smoothing import apply_pose_smoothing


def smooth_method_channel_id(method: str) -> int:
    return {"none": 0, "butterworth_sosfiltfilt": 1}.get(str(method), 0)


def compute_features_from_hml3d(
    poses: np.ndarray,
    dt: float,
    include_poses: bool,
    model_path: str | None,
    *,
    fps: int,
    sampling_frequency: float | None,
    smooth_poses: bool,
    smooth_cutoff_hz: float,
    smooth_butterworth_order: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if poses.ndim == 2 and poses.shape == (22, 3):
        poses = poses[None, ...]
    if poses.ndim != 3 or poses.shape[1:] != (22, 3):
        raise ValueError(f"Expected poses shape [T,22,3], got {poses.shape}")

    poses, smooth_meta = apply_pose_smoothing(
        poses.astype(np.float32),
        fps=int(fps),
        sampling_frequency=sampling_frequency,
        smooth_poses=bool(smooth_poses),
        smooth_cutoff_hz=float(smooth_cutoff_hz),
        smooth_butterworth_order=int(smooth_butterworth_order),
    )
    sid = smooth_method_channel_id(str(smooth_meta.get("method", "none")))
    cutoff_req = float(smooth_meta["cutoff_hz_requested"])
    cutoff_eff = smooth_meta["cutoff_hz_effective"]

    t = poses.shape[0]
    flat = poses.reshape(t, -1).astype(np.float32)
    velocities = finite_diff(flat, dt)
    accelerations = finite_diff(velocities, dt)
    torques = finite_diff(accelerations, dt)
    torque_rate = finite_diff(torques, dt)

    com_pos = np.mean(poses[:, [0, 1, 2, 3, 6, 9], :], axis=1).astype(np.float32)
    com_vel = finite_diff(com_pos, dt)
    com_acc = finite_diff(com_vel, dt)
    contact_feats = estimate_contact_from_hml3d(poses, com_acc, dt=dt)

    torque_power = np.sum(torques * velocities, axis=1, keepdims=True).astype(np.float32)
    power_pos = np.maximum(torque_power, 0.0).astype(np.float32)
    power_neg = np.minimum(torque_power, 0.0).astype(np.float32)
    feats: Dict[str, np.ndarray] = {
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
        "contact_inferred": np.array([1.0], dtype=np.float32),
        "contact_source_moco_estimated_proxy": np.array([1.0], dtype=np.float32),
        "inferred_contact_active_lr": contact_feats["inferred_contact_active_lr"],
        "inferred_contact_active": (np.sum(contact_feats["inferred_contact_active_lr"], axis=1, keepdims=True) > 0).astype(
            np.float32
        ),
        "inferred_contact_force_total": contact_feats["contact_wrench"][:, 3:6] + contact_feats["contact_wrench"][:, 9:12],
        "inferred_contact_wrench_lr": contact_feats["contact_wrench"],
        "torque_power": torque_power,
        "power_pos": power_pos,
        "power_neg": power_neg,
        "cum_pos_work": np.cumsum(power_pos, axis=0) * float(dt),
        "cum_neg_work": np.cumsum(power_neg, axis=0) * float(dt),
        "torque_l2": safe_norm(torques),
        "torque_rate_l2": safe_norm(torque_rate),
        "vel_l2": safe_norm(velocities),
        "acc_l2": safe_norm(accelerations),
        "jerk_l2": safe_norm(finite_diff(accelerations, dt)),
        "com_speed": safe_norm(com_vel, axis=1),
        "com_acc_l2": safe_norm(com_acc, axis=1),
        "com_jerk_l2": safe_norm(finite_diff(com_acc, dt), axis=1),
        "momentum_proxy_l2": safe_norm(velocities, axis=1),
        "ang_momentum_proxy_l2": safe_norm(flat * velocities, axis=1),
        "kinetic_proxy": 0.5 * np.sum(velocities * velocities, axis=1, keepdims=True).astype(np.float32),
        "effort_proxy_l1": np.sum(np.abs(torques), axis=1, keepdims=True).astype(np.float32),
        "velocity_spectral_ratio": fft_energy_ratio(velocities),
        "accel_spectral_ratio": fft_energy_ratio(accelerations),
        "torque_spectral_ratio": fft_energy_ratio(torques),
        "summary_mean_abs_tau": np.mean(np.abs(torques), axis=0, keepdims=True).astype(np.float32),
        "summary_std_tau": np.std(torques, axis=0, keepdims=True).astype(np.float32),
        "summary_mean_abs_vel": np.mean(np.abs(velocities), axis=0, keepdims=True).astype(np.float32),
        "summary_std_vel": np.std(velocities, axis=0, keepdims=True).astype(np.float32),
        "summary_contact_ratio": contact_feats["summary_contact_ratio"],
        "dt": np.array([dt], dtype=np.float32),
        "pose_smooth_enabled": np.array([1.0 if smooth_meta["enabled"] else 0.0], dtype=np.float32),
        "pose_smooth_method_id": np.array([float(sid)], dtype=np.float32),
        "pose_smooth_cutoff_hz_requested": np.array([cutoff_req], dtype=np.float32),
        "pose_smooth_cutoff_hz_effective": np.array(
            [float(cutoff_eff) if cutoff_eff is not None else 0.0], dtype=np.float32
        ),
        "pose_smooth_butterworth_order": np.array([float(smooth_meta.get("butterworth_order") or 0)], dtype=np.float32),
    }
    if include_poses:
        feats["positions"] = flat

    if model_path is not None:
        mp = Path(model_path)
        feats["model_path_hash32"] = np.array([int(sha256(mp)[:8], 16)], dtype=np.uint32) if mp.exists() else np.array(
            [0], dtype=np.uint32
        )
    return feats, smooth_meta
