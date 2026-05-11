"""Finite-difference kinematics and spectral summaries (Torch)."""

from __future__ import annotations

import torch


def finite_diff(x: torch.Tensor, dt: float) -> torch.Tensor:
    """Centered finite difference along time (axis=0)."""
    if x.shape[0] < 2:
        return torch.zeros_like(x)
    y = torch.zeros_like(x)
    y[0] = (x[1] - x[0]) / float(dt)
    y[-1] = (x[-1] - x[-2]) / float(dt)
    if x.shape[0] > 2:
        y[1:-1] = (x[2:] - x[:-2]) / (2.0 * float(dt))
    return y


def safe_norm(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return torch.linalg.norm(x, dim=dim, keepdim=True)


def fft_energy_ratio(x: torch.Tensor) -> torch.Tensor:
    """High-frequency energy ratio as ``[1, C]``."""
    if x.shape[0] < 4:
        return torch.zeros((1, x.shape[1]), dtype=x.dtype, device=x.device)
    fx = torch.fft.rfft(x, dim=0)
    power = fx.real.pow(2) + fx.imag.pow(2)
    n_freq = int(power.shape[0])
    split = max(1, n_freq // 4)
    low = power[:split].sum(dim=0)
    high = power[split:].sum(dim=0)
    ratio = high / torch.clamp(low + high, min=1e-8)
    return ratio.reshape(1, -1)
