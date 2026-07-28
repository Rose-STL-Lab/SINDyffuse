"""SINDy u/c features from MinT q (proxies)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


def features_from_mint_q(q: np.ndarray, fps: float) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Build u/c from MinT ``q`` ``[T, ndof]`` using pelvis translation proxies."""
    x = np.asarray(q, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected q [T, ndof], got {x.shape}")
    t_len = int(x.shape[0])
    dt = 1.0 / max(float(fps), 1e-8)
    root = x[:, : min(3, x.shape[1])]
    if root.shape[1] < 3:
        pad = np.zeros((t_len, 3 - root.shape[1]), dtype=np.float32)
        root = np.concatenate([root, pad], axis=1)
    if t_len > 1:
        root_v = np.gradient(root, dt, axis=0).astype(np.float32)
        root_a = np.gradient(root_v, dt, axis=0).astype(np.float32)
    else:
        root_v = np.zeros_like(root, dtype=np.float32)
        root_a = np.zeros_like(root, dtype=np.float32)

    phase = np.linspace(0.0, 2.0 * np.pi, num=t_len, endpoint=False, dtype=np.float32)
    cadence = np.full((t_len,), fill_value=(1.0 / max(t_len * dt, 1e-6)), dtype=np.float32)
    u_names = [
        "phase_sin",
        "phase_cos",
        "cadence_hz",
        "root_speed_x",
        "root_speed_y",
        "root_speed_z",
        "root_acc_x",
        "root_acc_y",
        "root_acc_z",
        "pelvis_height",
    ]
    u = np.stack(
        [
            np.sin(phase),
            np.cos(phase),
            cadence,
            root_v[:, 0],
            root_v[:, 1],
            root_v[:, 2],
            root_a[:, 0],
            root_a[:, 1],
            root_a[:, 2],
            root[:, 1],
        ],
        axis=-1,
    ).astype(np.float32)

    height = root[:, 1]
    ground = float(np.percentile(height, 2.0)) if t_len > 0 else 0.0
    speed = np.zeros((t_len,), dtype=np.float32)
    if t_len > 1:
        speed[1:] = np.linalg.norm(root[1:] - root[:-1], axis=-1)
    contact = ((height <= ground + 0.03) & (speed <= 0.05)).astype(np.float32)
    c_names = ["contact_left", "contact_right", "double_support", "single_support", "grf_trust_mask"]
    c = np.stack(
        [
            contact,
            contact,
            contact,
            1.0 - contact,
            np.ones((t_len,), dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)
    return u, c, u_names, c_names


def features_from_mint_q_torch(
    q: torch.Tensor, fps: float
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    """Torch u/c from MinT ``q`` ``[B, T, ndof]`` or ``[T, ndof]``."""
    batched = q.ndim == 3
    if not batched:
        q = q.unsqueeze(0)
    b, t_len, _ = q.shape
    dt = 1.0 / max(float(fps), 1e-8)
    root = q[:, :, : min(3, q.shape[2])]
    if root.shape[2] < 3:
        pad = torch.zeros(b, t_len, 3 - root.shape[2], device=q.device, dtype=q.dtype)
        root = torch.cat([root, pad], dim=2)
    if t_len > 1:
        root_v = torch.gradient(root, spacing=dt, dim=1)[0]
        root_a = torch.gradient(root_v, spacing=dt, dim=1)[0]
    else:
        root_v = torch.zeros_like(root)
        root_a = torch.zeros_like(root)

    phase = torch.linspace(0.0, 2.0 * 3.14159265, steps=t_len, device=q.device, dtype=q.dtype)
    phase = phase.view(1, t_len).expand(b, t_len)
    cadence = torch.full((b, t_len), fill_value=(1.0 / max(t_len * dt, 1e-6)), device=q.device, dtype=q.dtype)
    u_names = [
        "phase_sin",
        "phase_cos",
        "cadence_hz",
        "root_speed_x",
        "root_speed_y",
        "root_speed_z",
        "root_acc_x",
        "root_acc_y",
        "root_acc_z",
        "pelvis_height",
    ]
    u = torch.stack(
        [
            torch.sin(phase),
            torch.cos(phase),
            cadence,
            root_v[:, :, 0],
            root_v[:, :, 1],
            root_v[:, :, 2],
            root_a[:, :, 0],
            root_a[:, :, 1],
            root_a[:, :, 2],
            root[:, :, 1],
        ],
        dim=-1,
    )
    height = root[:, :, 1]
    ground = torch.quantile(height.reshape(-1), 0.02) if t_len > 0 else torch.tensor(0.0, device=q.device)
    speed = torch.zeros(b, t_len, device=q.device, dtype=q.dtype)
    if t_len > 1:
        speed[:, 1:] = torch.linalg.norm(root[:, 1:] - root[:, :-1], dim=-1)
    contact = ((height <= ground + 0.03) & (speed <= 0.05)).to(q.dtype)
    c_names = ["contact_left", "contact_right", "double_support", "single_support", "grf_trust_mask"]
    c = torch.stack(
        [
            contact,
            contact,
            contact,
            1.0 - contact,
            torch.ones(b, t_len, device=q.device, dtype=q.dtype),
        ],
        dim=-1,
    )
    if not batched:
        return u[0], c[0], u_names, c_names
    return u, c, u_names, c_names
