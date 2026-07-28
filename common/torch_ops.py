"""Finite-difference kinematics helpers (Torch)."""

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


__all__ = ["finite_diff", "safe_norm"]
