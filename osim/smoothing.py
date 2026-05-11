"""Pose smoothing before kinematic derivatives (Torch FIR + SciPy ``firwin`` kernel design)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as sp_signal  # type: ignore[import-untyped]


def sample_rate_hz(fps: int, sampling_frequency: float | None) -> float:
    return float(sampling_frequency if sampling_frequency is not None else fps)


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
    w_np = sp_signal.firwin(num_taps, eff_fc, fs=sr)
    w_t = torch.as_tensor(w_np, dtype=poses.dtype, device=poses.device).view(1, 1, -1)

    t_frames = int(poses.shape[0])
    x = poses.reshape(1, -1, t_frames)
    c = int(x.shape[1])
    weight = w_t.expand(c, 1, num_taps)
    pad = (num_taps - 1) // 2
    xpad = F.pad(x, (pad, pad), mode="reflect")
    y = F.conv1d(xpad, weight, bias=None, stride=1, padding=0, groups=c)
    smoothed = y.reshape(poses.shape)
    return smoothed, meta


def apply_pose_smoothing(
    poses: np.ndarray,
    *,
    fps: int,
    sampling_frequency: float | None,
    smooth_poses: bool,
    smooth_cutoff_hz: float,
    smooth_butterworth_order: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """NumPy API: delegates to :func:`apply_pose_smoothing_torch` then returns ``float32`` arrays."""
    t = torch.as_tensor(poses, dtype=torch.float32)
    out, meta = apply_pose_smoothing_torch(
        t,
        fps=int(fps),
        sampling_frequency=sampling_frequency,
        smooth_poses=bool(smooth_poses),
        smooth_cutoff_hz=float(smooth_cutoff_hz),
        smooth_butterworth_order=int(smooth_butterworth_order),
    )
    return out.detach().cpu().numpy().astype(np.float32), meta
