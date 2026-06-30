"""Pose smoothing before kinematic derivatives (Torch FIR + SciPy ``firwin`` kernel design)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as sp_signal  # type: ignore[import-untyped]


def sample_rate_hz(fps: int, sampling_frequency: float | None) -> float:
    return float(sampling_frequency if sampling_frequency is not None else fps)


# Rajagopal / OpenSim: DOFs 0–2 are pelvis rotations; 3–5 are translations; 6+ are joint angles.
_PELVIS_ROT_DOF_SLICE = slice(0, 3)
_PELVIS_TRANS_DOF_SLICE = slice(3, 6)


def unwrap_pose_angles(poses: np.ndarray) -> np.ndarray:
    """Remove ``2π`` jumps along time on rotational DOFs (not pelvis translations)."""
    x = np.asarray(poses, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        return x
    out = x.copy()
    if out.shape[1] > _PELVIS_ROT_DOF_SLICE.stop:
        out[:, _PELVIS_ROT_DOF_SLICE] = np.unwrap(
            out[:, _PELVIS_ROT_DOF_SLICE], axis=0
        )
    if out.shape[1] > _PELVIS_TRANS_DOF_SLICE.stop:
        out[:, _PELVIS_TRANS_DOF_SLICE.stop :] = np.unwrap(
            out[:, _PELVIS_TRANS_DOF_SLICE.stop :], axis=0
        )
    return out


def effective_pose_smooth_cutoff_hz(cutoff_hz: float, sample_rate_hz: float) -> float:
    nyq = 0.5 * sample_rate_hz
    cap = nyq * 0.499
    return float(np.clip(cutoff_hz, 0.1, max(cap - 1e-3, 0.25 * nyq)))


def _fir_num_taps(sample_rate_hz: float, cutoff_hz: float, butterworth_order: int) -> int:
    """Odd-length FIR tap count from sample rate, cutoff, and legacy order knob."""
    sr = float(sample_rate_hz)
    fc = float(max(cutoff_hz, 0.1))
    order = int(max(1, butterworth_order))
    base = 2 * int(sr / fc) + 1 + 2 * max(0, order - 2) * 5
    num_taps = min(401, max(5, base))
    if num_taps % 2 == 0:
        num_taps += 1
    return num_taps


def apply_pose_smoothing_torch(
    poses: torch.Tensor,
    *,
    fps: int,
    sampling_frequency: float | None,
    smooth_poses: bool,
    smooth_cutoff_hz: float,
    smooth_butterworth_order: int,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Low-pass along time via grouped ``conv1d`` (reflect padding, FIR from ``firwin``)."""
    sr = sample_rate_hz(fps, sampling_frequency)
    meta: Dict[str, Any] = {
        "enabled": bool(smooth_poses),
        "method": "none",
        "cutoff_hz_requested": float(smooth_cutoff_hz),
        "cutoff_hz_effective": None,
        "sample_rate_hz": float(sr),
        "butterworth_order": int(smooth_butterworth_order),
    }
    if not smooth_poses or poses.shape[0] < 2:
        meta["enabled"] = bool(smooth_poses and poses.shape[0] >= 2)
        return poses, meta

    eff_fc = effective_pose_smooth_cutoff_hz(smooth_cutoff_hz, sr)
    meta["cutoff_hz_effective"] = float(eff_fc)
    meta["method"] = "fir_firwin_conv1d"

    num_taps = _fir_num_taps(sr, eff_fc, smooth_butterworth_order)
    pad = (num_taps - 1) // 2
    t_frames = int(poses.shape[0])
    if t_frames <= 2 * pad:
        meta["enabled"] = False
        meta["method"] = "skipped_short_clip"
        return poses, meta

    w_np = sp_signal.firwin(num_taps, eff_fc, fs=sr)
    w_t = torch.as_tensor(w_np, dtype=poses.dtype, device=poses.device).view(1, 1, -1)

    # ``poses`` has time on axis 0; the rest (DOFs, keypoints*3, ...) are
    # treated as independent channels and convolved along time separately.
    # ``conv1d`` expects ``[B, C, L]`` with each channel on its own row and
    # time on the L axis. The natural memory layout of ``[T, ...]`` puts
    # different channels adjacent at each time step, so we MUST transpose
    # time to the L axis -- a raw ``reshape(1, -1, T)`` (the historical
    # implementation) interleaves channels into the time axis and silently
    # corrupts the signal (it scrambled pelvis q in every produced B3D file).
    orig_shape = tuple(poses.shape)
    # Flatten everything except time into a single channel dimension.
    x_flat = poses.reshape(t_frames, -1)
    n_channels = int(x_flat.shape[1])
    # Transpose [T, C] -> [C, T], add batch dim -> [1, C, T].
    x = x_flat.transpose(0, 1).contiguous().unsqueeze(0)
    weight = w_t.expand(n_channels, 1, num_taps)
    pad = (num_taps - 1) // 2
    xpad = F.pad(x, (pad, pad), mode="reflect")
    y = F.conv1d(xpad, weight, bias=None, stride=1, padding=0, groups=n_channels)
    # [1, C, T] -> [T, C] -> original shape.
    y = y.squeeze(0).transpose(0, 1).contiguous()
    smoothed = y.reshape(orig_shape)
    return smoothed, meta


def apply_pose_smoothing_numpy(
    poses: np.ndarray,
    *,
    fps: float,
    sampling_frequency: float | None = None,
    smooth_poses: bool = True,
    smooth_cutoff_hz: float = 6.0,
    smooth_butterworth_order: int = 2,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Low-pass ``poses`` along time; input shape ``(T, num_dofs)``."""
    x = np.asarray(poses, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected poses [T, D], got {x.shape}")
    x = unwrap_pose_angles(x.astype(np.float64)).astype(np.float32)
    t = torch.from_numpy(x)
    out, meta = apply_pose_smoothing_torch(
        t,
        fps=int(round(fps)),
        sampling_frequency=sampling_frequency,
        smooth_poses=bool(smooth_poses),
        smooth_cutoff_hz=float(smooth_cutoff_hz),
        smooth_butterworth_order=int(smooth_butterworth_order),
    )
    return out.numpy(), meta


def smooth_activation_trajectory(
    activations: np.ndarray,
    *,
    fps: float,
    cutoff_hz: float,
    butterworth_order: int = 2,
) -> np.ndarray:
    """Low-pass muscle activations along time; input ``[T, M]``."""
    act = np.asarray(activations, dtype=np.float64)
    if float(cutoff_hz) <= 0.0 or act.ndim != 2 or act.shape[0] < 3:
        return act.astype(np.float32)

    sr = float(fps)
    eff_fc = effective_pose_smooth_cutoff_hz(float(cutoff_hz), sr)
    num_taps = _fir_num_taps(sr, eff_fc, int(butterworth_order))
    if act.shape[0] <= num_taps:
        return act.astype(np.float32)

    kernel = sp_signal.firwin(num_taps, eff_fc, fs=sr)
    out = np.empty_like(act)
    for m in range(int(act.shape[1])):
        col = act[:, m]
        if not np.isfinite(col).all():
            col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
        out[:, m] = sp_signal.filtfilt(kernel, [1.0], col, method="gust")
    return np.clip(out, 0.0, 1.0).astype(np.float32)

