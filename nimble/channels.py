"""Differentiable biomechanical scalars for Nimble guidance (Torch, per frame ``[T, 1]``)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from nimble.contact import estimate_contact_from_feet
from nimble.ops import finite_diff, safe_norm
from nimble.rajagopal_kin import COM_KEYPOINT_INDICES, IDX_FOOT_L, IDX_FOOT_R

# All channels in L_bio (each is a per-frame magnitude / penalty scalar).
BIOMECH_COMPONENT_KEYS: Tuple[str, ...] = (
    # DOF (Rajagopal q)
    "vel",
    "acc",
    "torque",
    "torque_rate",
    "jerk",
    "effort",
    "joint_limit",
    "kinetic_q",
    "torque_power",
    # CoM (pose kinematics)
    "com_speed",
    "com_acc",
    "com_jerk",
    # Contact / GRF (pose + foot FK from q)
    "contact_gap",
    "contact_wrench",
    "grf_left",
    "grf_right",
    "grf_vertical",
    "grf_weight_deficit",
    "foot_slip",
    # Pose-space proxies
    "pose_vel",
    "pose_acc",
    "ang_momentum",
)

DEFAULT_BIOMECH_WEIGHT: float = 0.05


def default_biomech_weights(override: float | None = None) -> Dict[str, float]:
    w = float(DEFAULT_BIOMECH_WEIGHT if override is None else override)
    return {k: w for k in BIOMECH_COMPONENT_KEYS}


def weight_config_key(component: str) -> str:
    """JSON key under ``train.nimble_guidance.weights`` for a component."""
    if component == "contact_gap":
        return "lambda_contact"
    return f"lambda_{component}"


def parse_biomech_weights(wcfg: Dict[str, Any] | None) -> Dict[str, float]:
    """Build component weight dict from nested ``weights`` config."""
    out = default_biomech_weights()
    if not wcfg:
        return out
    for key in BIOMECH_COMPONENT_KEYS:
        cfg_key = weight_config_key(key)
        if cfg_key in wcfg:
            out[key] = float(wcfg[cfg_key])
    return out


class _BodyWorldOrigin(torch.autograd.Function):
    """World origin of a body node from ``q`` (linear FK + analytic Jacobian)."""

    @staticmethod
    def forward(ctx: Any, sk: Any, body_idx: int, q: torch.Tensor) -> torch.Tensor:
        q_np = q.detach().contiguous().cpu().numpy().astype(np.float64).reshape(-1)
        sk.setPositions(q_np)
        sk.computeForwardKinematics()
        tr = sk.getBodyNode(int(body_idx)).getWorldTransform()
        if hasattr(tr, "translation"):
            pos = np.asarray(tr.translation(), dtype=np.float64).reshape(3)
        else:
            pos = np.asarray(tr, dtype=np.float64).reshape(3)
        ctx.sk = sk
        ctx.body_idx = int(body_idx)
        ctx.save_for_backward(q)
        return torch.as_tensor(pos, dtype=q.dtype, device=q.device)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[None, None, torch.Tensor]:
        sk = ctx.sk
        body_idx = int(ctx.body_idx)
        (q,) = ctx.saved_tensors
        g = grad_out.detach().contiguous().cpu().numpy().astype(np.float64).reshape(3)
        q_np = q.detach().contiguous().cpu().numpy().astype(np.float64).reshape(-1)
        sk.setPositions(q_np)
        body = sk.getBodyNode(body_idx)
        off = np.zeros(3, dtype=np.float64)
        j = np.asarray(sk.getLinearJacobian(body, off), dtype=np.float64)
        grad_q = j.T @ g
        return None, None, torch.as_tensor(grad_q, dtype=q.dtype, device=q.device)


def _foot_positions_torch(
    sk: Any,
    foot_indices: Tuple[int, int],
    q_traj: torch.Tensor,
    *,
    use_torch_fk: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_torch_fk:
        from nimble.fk_torch import body_origins

        return body_origins(q_traj, foot_indices)
    left_idx, right_idx = foot_indices
    rows_l: List[torch.Tensor] = []
    rows_r: List[torch.Tensor] = []
    for t in range(int(q_traj.shape[0])):
        rows_l.append(_BodyWorldOrigin.apply(sk, left_idx, q_traj[t]))
        rows_r.append(_BodyWorldOrigin.apply(sk, right_idx, q_traj[t]))
    return torch.stack(rows_l, dim=0), torch.stack(rows_r, dim=0)


def _sigmoid_gate(x: torch.Tensor, center: float, sharpness: float) -> torch.Tensor:
    return torch.sigmoid((float(center) - x) * float(sharpness))


def _estimate_grf_lr_torch(
    foot_left: torch.Tensor,
    foot_right: torch.Tensor,
    com_acc: torch.Tensor,
    *,
    mass_kg: float,
    g_mps2: float,
    height_thresh_m: float,
    speed_thresh_mps: float,
    dt: float,
    gate_sharpness: float = 15.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left_vel = finite_diff(foot_left, float(dt))
    right_vel = finite_diff(foot_right, float(dt))
    ground_y = torch.minimum(foot_left[:, 1], foot_right[:, 1]).min()
    left_h = foot_left[:, 1] - ground_y
    right_h = foot_right[:, 1] - ground_y
    left_spd = torch.linalg.norm(left_vel[:, [0, 2]], dim=1)
    right_spd = torch.linalg.norm(right_vel[:, [0, 2]], dim=1)
    l_con = _sigmoid_gate(left_h, height_thresh_m, gate_sharpness) * _sigmoid_gate(
        left_spd, speed_thresh_mps, gate_sharpness
    )
    r_con = _sigmoid_gate(right_h, height_thresh_m, gate_sharpness) * _sigmoid_gate(
        right_spd, speed_thresh_mps, gate_sharpness
    )
    any_c = torch.clamp(l_con + r_con, max=1.0).unsqueeze(1)
    g_vec = torch.tensor([0.0, -float(g_mps2), 0.0], dtype=foot_left.dtype, device=foot_left.device)
    f_total = float(mass_kg) * (com_acc - g_vec.view(1, 3))
    f_total = f_total * any_c
    f_y = torch.relu(f_total[:, 1:2])
    f_total = torch.cat([f_total[:, 0:1], f_y, f_total[:, 2:3]], dim=1)
    denom = torch.clamp(l_con.unsqueeze(1) + r_con.unsqueeze(1), min=1e-3)
    f_l = f_total * l_con.unsqueeze(1) / denom
    f_r = f_total * r_con.unsqueeze(1) / denom
    return f_l, f_r, f_total[:, 1:2]


def build_channels(
    keypoints: torch.Tensor,
    fq: torch.Tensor,
    *,
    sk: Any,
    foot_indices: Tuple[int, int],
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    dt: float,
    guidance_cfg: Any,
    use_torch_fk: bool = False,
) -> Dict[str, torch.Tensor]:
    """Differentiable biomechanical scalars ``[T, 1]`` from Rajagopal keypoints and ``q``."""
    mass_kg = float(guidance_cfg.mass_kg)
    g_mps2 = float(guidance_cfg.g_mps2)
    h_thresh = float(guidance_cfg.contact_height_thresh_m)
    spd_thresh = float(guidance_cfg.contact_speed_thresh_mps)
    gate_sharp = float(getattr(guidance_cfg, "contact_gate_sharpness", 15.0))
    mg = float(mass_kg * g_mps2)

    qdot = finite_diff(fq, dt)
    qddot = finite_diff(qdot, dt)
    torque_vec = finite_diff(qddot, dt)
    torque_rate_vec = finite_diff(torque_vec, dt)

    vel = safe_norm(qdot)
    acc = safe_norm(qddot)
    torque = safe_norm(torque_vec)
    torque_rate = safe_norm(torque_rate_vec)
    jerk = safe_norm(finite_diff(torque_vec, dt))
    effort = torch.abs(torque_vec).sum(dim=1, keepdim=True)
    kinetic_q = 0.5 * (qdot * qdot).sum(dim=1, keepdim=True)
    torque_power = torch.abs((torque_vec * qdot).sum(dim=1, keepdim=True))

    lo = torch.as_tensor(q_lo, dtype=fq.dtype, device=fq.device).view(1, -1)
    hi = torch.as_tensor(q_hi, dtype=fq.dtype, device=fq.device).view(1, -1)
    over = torch.relu(fq - hi) + torch.relu(lo - fq)
    joint_limit = safe_norm(over, dim=1)

    com_idx = list(COM_KEYPOINT_INDICES)
    com_pos = keypoints[:, com_idx, :].mean(dim=1)
    com_vel = finite_diff(com_pos, dt)
    com_acc = finite_diff(com_vel, dt)
    com_speed = safe_norm(com_vel, dim=1)
    com_acc_l2 = safe_norm(com_acc, dim=1)
    com_jerk = safe_norm(finite_diff(com_acc, dt), dim=1)

    foot_left = keypoints[:, IDX_FOOT_L, :]
    foot_right = keypoints[:, IDX_FOOT_R, :]
    contact_feats = estimate_contact_from_feet(
        foot_left,
        foot_right,
        com_acc,
        dt=float(dt),
        mass_kg=mass_kg,
        g_mps2=g_mps2,
        height_thresh_m=h_thresh,
        speed_thresh_mps=spd_thresh,
        gate_sharpness=gate_sharp,
    )
    contact_wrench = contact_feats["contact_wrench_l2"]
    kin_gap = 1.0 - contact_feats["inferred_contact_active_lr"][:, :2].mean(dim=1, keepdim=True)

    foot_l, foot_r = _foot_positions_torch(sk, foot_indices, fq, use_torch_fk=use_torch_fk)
    f_l, f_r, f_vert = _estimate_grf_lr_torch(
        foot_l,
        foot_r,
        com_acc,
        mass_kg=mass_kg,
        g_mps2=g_mps2,
        height_thresh_m=h_thresh,
        speed_thresh_mps=spd_thresh,
        dt=float(dt),
        gate_sharpness=gate_sharp,
    )
    grf_left = safe_norm(f_l, dim=1)
    grf_right = safe_norm(f_r, dim=1)
    grf_vertical = f_vert
    grf_weight_deficit = torch.abs(grf_vertical - mg)
    physics_gap = 1.0 - torch.clamp((grf_left + grf_right) / max(0.03 * mg, 1e-3), min=0.0, max=1.0)
    contact_gap = 0.5 * (kin_gap + physics_gap)

    left_vel = finite_diff(foot_l, float(dt))
    right_vel = finite_diff(foot_r, float(dt))
    foot_slip = contact_feats["inferred_contact_active_lr"][:, 0:1] * torch.linalg.norm(
        left_vel[:, [0, 2]], dim=1, keepdim=True
    ) + contact_feats["inferred_contact_active_lr"][:, 1:2] * torch.linalg.norm(
        right_vel[:, [0, 2]], dim=1, keepdim=True
    )

    flat = keypoints.reshape(int(keypoints.shape[0]), -1)
    pose_vel = safe_norm(finite_diff(flat, dt))
    pose_acc = safe_norm(finite_diff(finite_diff(flat, dt), dt))
    ang_momentum = safe_norm(flat * finite_diff(flat, dt), dim=1)

    return {
        "vel": vel,
        "acc": acc,
        "torque": torque,
        "torque_rate": torque_rate,
        "jerk": jerk,
        "effort": effort,
        "joint_limit": joint_limit,
        "kinetic_q": kinetic_q,
        "torque_power": torque_power,
        "com_speed": com_speed,
        "com_acc": com_acc_l2,
        "com_jerk": com_jerk,
        "contact_gap": contact_gap,
        "contact_wrench": contact_wrench,
        "grf_left": grf_left,
        "grf_right": grf_right,
        "grf_vertical": grf_vertical,
        "grf_weight_deficit": grf_weight_deficit,
        "foot_slip": foot_slip,
        "pose_vel": pose_vel,
        "pose_acc": pose_acc,
        "ang_momentum": ang_momentum,
    }
