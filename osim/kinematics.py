"""Finite-difference kinematics helpers and spectral summaries."""

from __future__ import annotations

import numpy as np


def finite_diff(x: np.ndarray, dt: float) -> np.ndarray:
    if x.shape[0] < 2:
        return np.zeros_like(x, dtype=np.float32)
    return np.gradient(x.astype(np.float32), dt, axis=0).astype(np.float32)


def safe_norm(x: np.ndarray, axis: int = 1) -> np.ndarray:
    return np.linalg.norm(x.astype(np.float32), axis=axis, keepdims=True).astype(np.float32)


def fft_energy_ratio(x: np.ndarray) -> np.ndarray:
    if x.shape[0] < 4:
        return np.zeros((1, x.shape[1]), dtype=np.float32)
    fx = np.fft.rfft(x.astype(np.float32), axis=0)
    power = (np.abs(fx) ** 2).astype(np.float32)
    n_freq = power.shape[0]
    split = max(1, n_freq // 4)
    low = np.sum(power[:split, :], axis=0)
    high = np.sum(power[split:, :], axis=0)
    ratio = high / np.clip(low + high, 1e-8, None)
    return ratio.reshape(1, -1).astype(np.float32)
