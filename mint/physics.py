"""Generic q-space bio channels for MinT skeleton (Phase 3a proxies)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from nimble.channels import BIOMECH_COMPONENT_KEYS
from nimble.guidance import NimbleGuidanceConfig
from nimble.ops import finite_diff, safe_norm


def _fd(x: torch.Tensor, dt: float) -> torch.Tensor:
    return finite_diff(x, dt)


def _fd2(x: torch.Tensor, dt: float) -> torch.Tensor:
    return _fd(_fd(x, dt), dt)


def _fd3(x: torch.Tensor, dt: float) -> torch.Tensor:
    return _fd(_fd2(x, dt), dt)


def _scalar_norm(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Per-frame scalar magnitude ``[..., 1]``."""
    return safe_norm(x, dim=dim)


def _bio_components(x: torch.Tensor, dt: float) -> Dict[str, torch.Tensor]:
    """Shared proxy bio channels from ``q`` with time on the last-but-one or sole time axis."""
    vel = _fd(x, dt)
    acc = _fd2(x, dt)
    jerk = _fd3(x, dt)

    torque = acc
    torque_rate = _fd(torque, dt)
    effort = _scalar_norm(torque)
    joint_limit = torch.relu(torch.abs(x) - 2.5)
    kinetic_q = 0.5 * (vel ** 2)
    torque_power = (torque * vel).abs()

    com = x[..., : min(3, x.shape[-1])].mean(dim=-1, keepdim=True)
    com_speed = _fd(com, dt)
    com_acc = _fd2(com, dt)
    com_jerk = _fd3(com, dt)

    foot_l = x[..., min(7, x.shape[-1] - 1) : min(8, x.shape[-1])]
    foot_r = x[..., min(10, x.shape[-1] - 1) : min(11, x.shape[-1])]
    contact_gap = torch.relu(foot_l[..., :1] if foot_l.numel() else com * 0)
    contact_wrench = _scalar_norm(acc[..., : min(3, acc.shape[-1])])
    grf_left = _scalar_norm(foot_l) if foot_l.numel() else contact_wrench * 0
    grf_right = _scalar_norm(foot_r) if foot_r.numel() else contact_wrench * 0
    grf_vertical = grf_left + grf_right
    grf_weight_deficit = torch.relu(0.5 - grf_vertical)
    foot_slip = _scalar_norm(vel[..., : min(3, vel.shape[-1])])

    return {
        "vel": _scalar_norm(vel),
        "acc": _scalar_norm(acc),
        "torque": _scalar_norm(torque),
        "torque_rate": _scalar_norm(torque_rate),
        "jerk": _scalar_norm(jerk),
        "effort": _scalar_norm(effort),
        "joint_limit": _scalar_norm(joint_limit),
        "kinetic_q": _scalar_norm(kinetic_q),
        "torque_power": _scalar_norm(torque_power),
        "com_speed": _scalar_norm(com_speed),
        "com_acc": _scalar_norm(com_acc),
        "com_jerk": _scalar_norm(com_jerk),
        "contact_gap": contact_gap if contact_gap.numel() else torch.zeros_like(com_speed),
        "contact_wrench": contact_wrench,
        "grf_left": grf_left if grf_left.numel() else torch.zeros_like(com_speed),
        "grf_right": grf_right if grf_right.numel() else torch.zeros_like(com_speed),
        "grf_vertical": grf_vertical if grf_vertical.numel() else torch.zeros_like(com_speed),
        "grf_weight_deficit": grf_weight_deficit if grf_weight_deficit.numel() else torch.zeros_like(com_speed),
        "foot_slip": foot_slip,
        "pose_vel": _scalar_norm(vel),
        "pose_acc": _scalar_norm(acc),
        "ang_momentum": _scalar_norm(torque),
    }


def bio_matrix_mint(
    q: np.ndarray,
    *,
    fps: float = 20.0,
    guidance_cfg: Optional[NimbleGuidanceConfig] = None,
) -> np.ndarray:
    """MinT ``q`` ``[T, ndof]`` → L_bio-style channels ``[T, C]`` via finite differences."""
    if q.ndim != 2:
        raise ValueError(f"Expected q [T, ndof], got {q.shape}")
    t_len = int(q.shape[0])
    dt = 1.0 / max(float(fps), 1e-8)
    x = torch.from_numpy(q.astype(np.float32))
    comp = _bio_components(x, dt)

    cols = []
    for key in BIOMECH_COMPONENT_KEYS:
        v = comp.get(key)
        if v is None:
            cols.append(np.zeros((t_len, 1), dtype=np.float32))
        else:
            cols.append(v.reshape(t_len, 1).detach().cpu().numpy().astype(np.float32))
    bio = np.concatenate(cols, axis=-1).astype(np.float32)
    return bio[:t_len]


def bio_from_q_torch(q: torch.Tensor, *, fps: float) -> torch.Tensor:
    """MinT ``q`` ``[B, T, ndof]`` → L_bio ``[B, T, C]`` (differentiable proxies)."""
    if q.ndim != 3:
        raise ValueError(f"Expected q [B, T, ndof], got {q.shape}")
    b, t_len, _ = q.shape
    dt = 1.0 / max(float(fps), 1e-8)
    rows = []
    for i in range(b):
        comp = _bio_components(q[i], dt)
        cols = [comp[key].reshape(t_len, 1) for key in BIOMECH_COMPONENT_KEYS]
        rows.append(torch.cat(cols, dim=-1))
    return torch.stack(rows, dim=0)
