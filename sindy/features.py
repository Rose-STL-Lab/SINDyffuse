"""Root and contact features (u, c) for the SINDy θ library — Rajagopal q-space only."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import torch

from nimble.rajagopal_kin import IDX_FOOT_L, IDX_FOOT_R, IDX_PELVIS, keypoints_numpy, keypoints_torch


def features_from_keypoints(keypoints: np.ndarray, fps: float) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Build u/c from Rajagopal keypoints ``[T, K, 3]``."""
    kp = np.asarray(keypoints, dtype=np.float32)
    if kp.ndim != 3 or kp.shape[1] < 6:
        raise ValueError(f"Expected keypoints [T, K>=6, 3], got {kp.shape}")
    t = int(kp.shape[0])
    dt = 1.0 / max(float(fps), 1e-8)
    root = kp[:, IDX_PELVIS, :]
    if t > 1:
        root_v = np.gradient(root, dt, axis=0).astype(np.float32)
        root_a = np.gradient(root_v, dt, axis=0).astype(np.float32)
    else:
        root_v = np.zeros_like(root, dtype=np.float32)
        root_a = np.zeros_like(root, dtype=np.float32)

    phase = np.linspace(0.0, 2.0 * np.pi, num=t, endpoint=False, dtype=np.float32)
    cadence = np.full((t,), fill_value=(1.0 / max(t * dt, 1e-6)), dtype=np.float32)
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

    heights = kp[:, :, 1]
    ground = float(np.percentile(heights, 2.0))

    def _contact(foot_idx: int) -> np.ndarray:
        pos = kp[:, foot_idx, :]
        speed = np.zeros((t,), dtype=np.float32)
        if t > 1:
            speed[1:] = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)
        return ((pos[:, 1] <= ground + 0.03) & (speed <= 0.02)).astype(np.float32)

    c_l, c_r = _contact(IDX_FOOT_L), _contact(IDX_FOOT_R)
    c_names = ["contact_left", "contact_right", "double_support", "single_support", "grf_trust_mask"]
    double_support = ((c_l > 0.5) & (c_r > 0.5)).astype(np.float32)
    single_support = ((c_l > 0.5) ^ (c_r > 0.5)).astype(np.float32)
    trust = np.ones((t,), dtype=np.float32)
    c = np.stack([c_l, c_r, double_support, single_support, trust], axis=-1).astype(np.float32)
    return u, c, u_names, c_names


def features_from_keypoints_torch(
    keypoints: torch.Tensor, fps: float
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    b, t, _, _ = keypoints.shape
    dt = 1.0 / max(float(fps), 1e-8)
    root = keypoints[:, :, IDX_PELVIS, :]
    root_v = torch.zeros_like(root)
    root_a = torch.zeros_like(root)
    if t > 1:
        root_v[:, 1:, :] = (root[:, 1:, :] - root[:, :-1, :]) / float(dt)
        root_a[:, 1:, :] = (root_v[:, 1:, :] - root_v[:, :-1, :]) / float(dt)

    phase = torch.linspace(0.0, 2.0 * torch.pi, steps=t, device=keypoints.device, dtype=keypoints.dtype)
    cadence_value = 1.0 / max(t * dt, 1e-6)
    cadence = torch.full((b, t), cadence_value, device=keypoints.device, dtype=keypoints.dtype)
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
            torch.sin(phase).unsqueeze(0).expand(b, t),
            torch.cos(phase).unsqueeze(0).expand(b, t),
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

    pos_l = keypoints[:, :, IDX_FOOT_L, :]
    pos_r = keypoints[:, :, IDX_FOOT_R, :]
    speed = torch.zeros((b, t), device=keypoints.device, dtype=keypoints.dtype)
    if t > 1:
        speed[:, 1:] = torch.linalg.norm(pos_l[:, 1:, :] - pos_l[:, :-1, :], dim=-1)

    heights = keypoints[:, :, :, 1]
    flat_heights = heights.reshape(b, -1)
    q = int(max(1, min(flat_heights.numel(), int(flat_heights.numel() * 0.02))))
    ground = torch.kthvalue(flat_heights, q, dim=1).values

    c_l = ((pos_l[:, :, 1] <= ground.unsqueeze(1) + 0.03) & (speed <= 0.02)).to(keypoints.dtype)
    c_r = ((pos_r[:, :, 1] <= ground.unsqueeze(1) + 0.03) & (speed <= 0.02)).to(keypoints.dtype)
    c_names = ["contact_left", "contact_right", "double_support", "single_support", "grf_trust_mask"]
    double_support = ((c_l > 0.5) & (c_r > 0.5)).to(keypoints.dtype)
    single_support = ((c_l > 0.5) ^ (c_r > 0.5)).to(keypoints.dtype)
    trust = torch.ones((b, t), device=keypoints.device, dtype=keypoints.dtype)
    c = torch.stack([c_l, c_r, double_support, single_support, trust], dim=-1)
    return u, c, u_names, c_names


def features_from_q(q: np.ndarray, sk: Any, fps: float) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Build u/c from Rajagopal ``q`` ``[T, ndof]``."""
    kp = keypoints_numpy(sk, q)
    return features_from_keypoints(kp, fps=fps)


def features_from_q_torch(
    q: torch.Tensor,
    sk: Any,
    fps: float,
    *,
    use_torch_fk: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    """Batch u/c from ``q`` ``[B, T, ndof]``."""
    b, _, _ = q.shape
    rows = []
    fk = bool(use_torch_fk and q.device.type != "cpu")
    for i in range(b):
        if fk:
            kp = keypoints_torch(q[i])
        else:
            kp = torch.from_numpy(keypoints_numpy(sk, q[i].detach().cpu().numpy()))
            kp = kp.to(device=q.device, dtype=q.dtype)
        rows.append(kp.unsqueeze(0))
    kp_b = torch.cat(rows, dim=0)
    return features_from_keypoints_torch(kp_b, fps=fps)
