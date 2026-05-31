"""Differentiable kinematic contact-force proxies from Rajagopal foot keypoints (Torch)."""

from __future__ import annotations

from typing import Dict

import torch

from nimble.ops import finite_diff, safe_norm


def _sigmoid_gate(x: torch.Tensor, center: float, sharpness: float) -> torch.Tensor:
    return torch.sigmoid((float(center) - x) * float(sharpness))


def estimate_contact_from_feet(
    foot_left: torch.Tensor,
    foot_right: torch.Tensor,
    com_acc: torch.Tensor,
    dt: float,
    mass_kg: float = 70.0,
    g_mps2: float = 9.81,
    height_thresh_m: float = 0.06,
    speed_thresh_mps: float = 1.2,
    gate_sharpness: float = 15.0,
) -> Dict[str, torch.Tensor]:
    """Continuous contact channels and wrench proxy (differentiable)."""
    left_vel = finite_diff(foot_left, dt)
    right_vel = finite_diff(foot_right, dt)

    ground_y = torch.minimum(foot_left[:, 1], foot_right[:, 1]).min()
    left_height = foot_left[:, 1] - ground_y
    right_height = foot_right[:, 1] - ground_y
    left_speed_xz = torch.linalg.norm(left_vel[:, [0, 2]], dim=1)
    right_speed_xz = torch.linalg.norm(right_vel[:, [0, 2]], dim=1)

    l_contact = _sigmoid_gate(left_height, height_thresh_m, gate_sharpness) * _sigmoid_gate(
        left_speed_xz, speed_thresh_mps, gate_sharpness
    )
    r_contact = _sigmoid_gate(right_height, height_thresh_m, gate_sharpness) * _sigmoid_gate(
        right_speed_xz, speed_thresh_mps, gate_sharpness
    )
    contact_lr = torch.stack([l_contact, r_contact], dim=1)
    any_contact = torch.clamp(contact_lr.sum(dim=1, keepdim=True), max=1.0)

    g_vec = torch.tensor([0.0, -float(g_mps2), 0.0], dtype=foot_left.dtype, device=foot_left.device)
    f_total = float(mass_kg) * (com_acc - g_vec.view(1, 3))
    f_y = torch.clamp(f_total[:, 1:2], min=0.0)
    f_total = torch.cat([f_total[:, 0:1], f_y, f_total[:, 2:3]], dim=1) * any_contact
    f_l = f_total * contact_lr[:, 0:1]
    f_r = f_total * contact_lr[:, 1:2]
    denom = torch.clamp(contact_lr[:, 0:1] + contact_lr[:, 1:2], min=1e-3)
    f_l = f_l / denom
    f_r = f_r / denom
    tau_l = torch.zeros_like(f_l)
    tau_r = torch.zeros_like(f_r)
    wrench = torch.cat([tau_l, f_l, tau_r, f_r], dim=1)
    wrench_l2 = safe_norm(wrench)
    return {
        "contact_wrench": wrench,
        "contact_wrench_l2": wrench_l2,
        "contact_active": torch.clamp(wrench_l2 * 10.0, max=1.0),
        "summary_contact_ratio": contact_lr.mean(dim=0, keepdim=True),
        "inferred_contact_active_lr": contact_lr,
    }
