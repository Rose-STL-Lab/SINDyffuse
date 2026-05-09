"""Zero-phase Butterworth pose smoothing before kinematic derivatives."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy import signal as sp_signal  # type: ignore[import-untyped]


def sample_rate_hz(fps: int, sampling_frequency: float | None) -> float:
    return float(sampling_frequency if sampling_frequency is not None else fps)


def smooth_poses_butterworth_sosfiltfilt(
    poses: np.ndarray,
    *,
    sample_rate_hz: float,
    cutoff_hz_effective: float,
    order: int,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass along time (SciPy ``sosfiltfilt``)."""
    fs = float(sample_rate_hz)
    fn = float(np.clip(cutoff_hz_effective, 0.1, fs * 0.499))
    wn = fn / (0.5 * fs)
    sos = sp_signal.butter(int(order), wn, btype="low", output="sos")
    out = poses.copy().astype(np.float64)
    for j in range(poses.shape[1]):
        for c in range(3):
            out[:, j, c] = sp_signal.sosfiltfilt(sos, poses[:, j, c].astype(np.float64))
    return out.astype(np.float32)


def effective_pose_smooth_cutoff_hz(cutoff_hz: float, sample_rate_hz: float) -> float:
    nyq = 0.5 * sample_rate_hz
    cap = nyq * 0.499
    return float(np.clip(cutoff_hz, 0.1, max(cap - 1e-3, 0.25 * nyq)))


def apply_pose_smoothing(
    poses: np.ndarray,
    *,
    fps: int,
    sampling_frequency: float | None,
    smooth_poses: bool,
    smooth_cutoff_hz: float,
    smooth_butterworth_order: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Optional Butterworth + ``sosfiltfilt`` on ``[T,22,3]`` joint positions."""
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
        return poses.astype(np.float32), meta

    eff_fc = effective_pose_smooth_cutoff_hz(smooth_cutoff_hz, sr)
    meta["cutoff_hz_effective"] = float(eff_fc)
    meta["method"] = "butterworth_sosfiltfilt"

    smoothed = smooth_poses_butterworth_sosfiltfilt(
        poses.astype(np.float32),
        sample_rate_hz=sr,
        cutoff_hz_effective=eff_fc,
        order=int(smooth_butterworth_order),
    )
    return smoothed, meta
