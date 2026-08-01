from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from nimble.contact import estimate_contact_from_feet
from nimble.dof_groups import LIMIT_GROUP_SPECS, SYMMETRY_PAIRS
from nimble.inverse_dynamics import id_group_norms, id_power_total, id_symmetry_scalars, joint_torques_torch
from nimble.ops import finite_diff, safe_norm
from nimble.rajagopal_kin import COM_KEYPOINT_INDICES, IDX_FOOT_L, IDX_FOOT_R
CONTACT_CHANNEL_KEYS: Tuple[str, ...] = ('contact_gap', 'contact_wrench', 'grf_left', 'grf_right', 'grf_vertical', 'grf_weight_deficit', 'foot_slip')
ID_CHANNEL_KEYS: Tuple[str, ...] = ('id_pelvis', 'id_hip_r', 'id_leg_r', 'id_hip_l', 'id_leg_l', 'id_lumbar', 'id_arm_r', 'id_arm_l')
LIMIT_CHANNEL_KEYS: Tuple[str, ...] = tuple((name for name, _ in LIMIT_GROUP_SPECS))
GAIT_CHANNEL_KEYS: Tuple[str, ...] = ('sym_hip_flex', 'sym_hip_add', 'sym_hip_rot', 'sym_knee', 'sym_ankle', 'sym_mtp', 'foot_clearance_l', 'foot_clearance_r')
DERIVED_CHANNEL_KEYS: Tuple[str, ...] = ('id_power_total', 'id_sym_hip', 'id_sym_knee', 'id_sym_ankle', 'grf_asymmetry', 'contact_power', 'limit_margin_min', 'support_width')
BIOMECH_COMPONENT_KEYS: Tuple[str, ...] = CONTACT_CHANNEL_KEYS + ID_CHANNEL_KEYS + LIMIT_CHANNEL_KEYS + GAIT_CHANNEL_KEYS + DERIVED_CHANNEL_KEYS
L_BIO_SCHEMA_VERSION = 2
DEFAULT_BIOMECH_WEIGHT: float = 0.05
DEFAULT_ID_BIOMECH_WEIGHT: float = 0.02

def default_biomech_weights(override: float | None=None) -> Dict[str, float]:
    w = float(DEFAULT_BIOMECH_WEIGHT if override is None else override)
    out = {k: w for k in BIOMECH_COMPONENT_KEYS}
    for key in ID_CHANNEL_KEYS:
        out[key] = float(DEFAULT_ID_BIOMECH_WEIGHT)
    for key in DERIVED_CHANNEL_KEYS:
        if key.startswith('id_'):
            out[key] = float(DEFAULT_ID_BIOMECH_WEIGHT)
    return out

def weight_config_key(component: str) -> str:
    if component == 'contact_gap':
        return 'lambda_contact'
    return f'lambda_{component}'

def parse_biomech_weights(wcfg: Dict[str, Any] | None) -> Dict[str, float]:
    out = default_biomech_weights()
    if not wcfg:
        return out
    for key in BIOMECH_COMPONENT_KEYS:
        cfg_key = weight_config_key(key)
        if cfg_key in wcfg:
            out[key] = float(wcfg[cfg_key])
    return out

class _BodyWorldOrigin(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, sk: Any, body_idx: int, q: torch.Tensor) -> torch.Tensor:
        q_np = q.detach().contiguous().cpu().numpy().astype(np.float64).reshape(-1)
        sk.setPositions(q_np)
        sk.computeForwardKinematics()
        tr = sk.getBodyNode(int(body_idx)).getWorldTransform()
        if hasattr(tr, 'translation'):
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
        q, = ctx.saved_tensors
        g = grad_out.detach().contiguous().cpu().numpy().astype(np.float64).reshape(3)
        q_np = q.detach().contiguous().cpu().numpy().astype(np.float64).reshape(-1)
        sk.setPositions(q_np)
        body = sk.getBodyNode(body_idx)
        off = np.zeros(3, dtype=np.float64)
        j = np.asarray(sk.getLinearJacobian(body, off), dtype=np.float64)
        grad_q = j.T @ g
        return (None, None, torch.as_tensor(grad_q, dtype=q.dtype, device=q.device))

def _foot_positions_torch(sk: Any, foot_indices: Tuple[int, int], q_traj: torch.Tensor, *, use_torch_fk: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_torch_fk:
        from nimble.fk_torch import body_origins
        return body_origins(q_traj, foot_indices)
    left_idx, right_idx = foot_indices
    rows_l: List[torch.Tensor] = []
    rows_r: List[torch.Tensor] = []
    for t in range(int(q_traj.shape[0])):
        rows_l.append(_BodyWorldOrigin.apply(sk, left_idx, q_traj[t]))
        rows_r.append(_BodyWorldOrigin.apply(sk, right_idx, q_traj[t]))
    return (torch.stack(rows_l, dim=0), torch.stack(rows_r, dim=0))

def _sigmoid_gate(x: torch.Tensor, center: float, sharpness: float) -> torch.Tensor:
    return torch.sigmoid((float(center) - x) * float(sharpness))

def build_limit_channels(fq: torch.Tensor, q_lo: np.ndarray, q_hi: np.ndarray) -> Dict[str, torch.Tensor]:
    lo = torch.as_tensor(q_lo, dtype=fq.dtype, device=fq.device).view(1, -1)
    hi = torch.as_tensor(q_hi, dtype=fq.dtype, device=fq.device).view(1, -1)
    out: Dict[str, torch.Tensor] = {}
    for name, sl in LIMIT_GROUP_SPECS:
        chunk = fq[:, sl]
        lo_g = lo[:, sl]
        hi_g = hi[:, sl]
        over = torch.relu(chunk - hi_g) + torch.relu(lo_g - chunk)
        out[name] = safe_norm(over, dim=1)
    return out

def limit_margin_min_scalar(fq: torch.Tensor, q_lo: np.ndarray, q_hi: np.ndarray) -> torch.Tensor:
    lo = torch.as_tensor(q_lo, dtype=fq.dtype, device=fq.device).view(1, -1)
    hi = torch.as_tensor(q_hi, dtype=fq.dtype, device=fq.device).view(1, -1)
    margin_lo = fq - lo
    margin_hi = hi - fq
    margin = torch.minimum(margin_lo, margin_hi)
    return margin.min(dim=1, keepdim=True).values

def build_symmetry_channels(fq: torch.Tensor) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, idx_l, idx_r in SYMMETRY_PAIRS:
        out[name] = torch.abs(fq[:, idx_l:idx_l + 1] - fq[:, idx_r:idx_r + 1])
    return out

def build_clearance_channels(keypoints: torch.Tensor) -> Dict[str, torch.Tensor]:
    foot_left = keypoints[:, IDX_FOOT_L, :]
    foot_right = keypoints[:, IDX_FOOT_R, :]
    ground_y = torch.minimum(foot_left[:, 1], foot_right[:, 1]).min()
    return {'foot_clearance_l': (foot_left[:, 1:2] - ground_y).clamp(min=0.0), 'foot_clearance_r': (foot_right[:, 1:2] - ground_y).clamp(min=0.0)}

def _estimate_grf_lr_torch(foot_left: torch.Tensor, foot_right: torch.Tensor, com_acc: torch.Tensor, *, mass_kg: float, g_mps2: float, height_thresh_m: float, speed_thresh_mps: float, dt: float, gate_sharpness: float=15.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    left_vel = finite_diff(foot_left, float(dt))
    right_vel = finite_diff(foot_right, float(dt))
    ground_y = torch.minimum(foot_left[:, 1], foot_right[:, 1]).min()
    left_h = foot_left[:, 1] - ground_y
    right_h = foot_right[:, 1] - ground_y
    left_spd = torch.linalg.norm(left_vel[:, [0, 2]], dim=1)
    right_spd = torch.linalg.norm(right_vel[:, [0, 2]], dim=1)
    l_con = _sigmoid_gate(left_h, height_thresh_m, gate_sharpness) * _sigmoid_gate(left_spd, speed_thresh_mps, gate_sharpness)
    r_con = _sigmoid_gate(right_h, height_thresh_m, gate_sharpness) * _sigmoid_gate(right_spd, speed_thresh_mps, gate_sharpness)
    contact_lr = torch.stack([l_con, r_con], dim=1)
    any_c = torch.clamp(l_con + r_con, max=1.0).unsqueeze(1)
    g_vec = torch.tensor([0.0, -float(g_mps2), 0.0], dtype=foot_left.dtype, device=foot_left.device)
    f_total = float(mass_kg) * (com_acc - g_vec.view(1, 3))
    f_total = f_total * any_c
    f_y = torch.relu(f_total[:, 1:2])
    f_total = torch.cat([f_total[:, 0:1], f_y, f_total[:, 2:3]], dim=1)
    denom = torch.clamp(contact_lr.sum(dim=1, keepdim=True), min=0.001)
    f_l = f_total * contact_lr[:, 0:1] / denom
    f_r = f_total * contact_lr[:, 1:2] / denom
    return (f_l, f_r, f_total[:, 1:2], contact_lr, left_vel, right_vel)

def build_contact_channels(keypoints: torch.Tensor, fq: torch.Tensor, *, sk: Any, foot_indices: Tuple[int, int], dt: float, guidance_cfg: Any, use_torch_fk: bool=False) -> Dict[str, torch.Tensor]:
    mass_kg = float(guidance_cfg.mass_kg)
    g_mps2 = float(guidance_cfg.g_mps2)
    h_thresh = float(guidance_cfg.contact_height_thresh_m)
    spd_thresh = float(guidance_cfg.contact_speed_thresh_mps)
    gate_sharp = float(getattr(guidance_cfg, 'contact_gate_sharpness', 15.0))
    mg = float(mass_kg * g_mps2)
    com_idx = list(COM_KEYPOINT_INDICES)
    com_pos = keypoints[:, com_idx, :].mean(dim=1)
    com_vel = finite_diff(com_pos, dt)
    com_acc = finite_diff(com_vel, dt)
    foot_left = keypoints[:, IDX_FOOT_L, :]
    foot_right = keypoints[:, IDX_FOOT_R, :]
    contact_feats = estimate_contact_from_feet(foot_left, foot_right, com_acc, dt=float(dt), mass_kg=mass_kg, g_mps2=g_mps2, height_thresh_m=h_thresh, speed_thresh_mps=spd_thresh, gate_sharpness=gate_sharp)
    contact_wrench = contact_feats['contact_wrench_l2']
    kin_gap = 1.0 - contact_feats['inferred_contact_active_lr'][:, :2].mean(dim=1, keepdim=True)
    foot_l, foot_r = _foot_positions_torch(sk, foot_indices, fq, use_torch_fk=use_torch_fk)
    f_l, f_r, f_vert, contact_lr, left_vel, right_vel = _estimate_grf_lr_torch(foot_l, foot_r, com_acc, mass_kg=mass_kg, g_mps2=g_mps2, height_thresh_m=h_thresh, speed_thresh_mps=spd_thresh, dt=float(dt), gate_sharpness=gate_sharp)
    grf_left = safe_norm(f_l, dim=1)
    grf_right = safe_norm(f_r, dim=1)
    grf_weight_deficit = torch.abs(f_vert - mg)
    physics_gap = 1.0 - torch.clamp((grf_left + grf_right) / max(0.03 * mg, 0.001), min=0.0, max=1.0)
    contact_gap = 0.5 * (kin_gap + physics_gap)
    foot_slip = contact_feats['inferred_contact_active_lr'][:, 0:1] * torch.linalg.norm(left_vel[:, [0, 2]], dim=1, keepdim=True) + contact_feats['inferred_contact_active_lr'][:, 1:2] * torch.linalg.norm(right_vel[:, [0, 2]], dim=1, keepdim=True)
    return {'contact_gap': contact_gap, 'contact_wrench': contact_wrench, 'grf_left': grf_left, 'grf_right': grf_right, 'grf_vertical': f_vert, 'grf_weight_deficit': grf_weight_deficit, 'foot_slip': foot_slip, '_f_l': f_l, '_f_r': f_r, '_contact_lr': contact_lr, '_left_vel': left_vel, '_right_vel': right_vel}

def build_id_channels(fq: torch.Tensor, *, sk: Any, foot_indices: Tuple[int, int], dt: float) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    tau, qdot = joint_torques_torch(fq, sk=sk, foot_indices=foot_indices, dt=float(dt))
    out = id_group_norms(tau)
    return (out, tau, qdot)

def build_derived_channels(*, tau: torch.Tensor, qdot: torch.Tensor, contact: Dict[str, torch.Tensor], fq: torch.Tensor, q_lo: np.ndarray, q_hi: np.ndarray, foot_l: torch.Tensor, foot_r: torch.Tensor) -> Dict[str, torch.Tensor]:
    f_l = contact['_f_l']
    f_r = contact['_f_r']
    contact_lr = contact['_contact_lr']
    left_vel = contact['_left_vel']
    right_vel = contact['_right_vel']
    grf_left = contact['grf_left']
    grf_right = contact['grf_right']
    denom = grf_left + grf_right + 0.001
    grf_asymmetry = torch.abs(grf_left - grf_right) / denom
    contact_power = contact_lr[:, 0:1] * (f_l * left_vel).sum(dim=1, keepdim=True).clamp(min=0.0) + contact_lr[:, 1:2] * (f_r * right_vel).sum(dim=1, keepdim=True).clamp(min=0.0)
    double_support = (contact_lr[:, 0] > 0.5) & (contact_lr[:, 1] > 0.5)
    lateral_sep = torch.linalg.norm(foot_l[:, [0, 2]] - foot_r[:, [0, 2]], dim=1, keepdim=True)
    support_width = lateral_sep * double_support.unsqueeze(1).to(lateral_sep.dtype)
    out: Dict[str, torch.Tensor] = {'id_power_total': id_power_total(tau, qdot), 'grf_asymmetry': grf_asymmetry, 'contact_power': contact_power, 'limit_margin_min': limit_margin_min_scalar(fq, q_lo, q_hi), 'support_width': support_width}
    out.update(id_symmetry_scalars(tau))
    return out

def build_channels(keypoints: torch.Tensor, fq: torch.Tensor, *, sk: Any, foot_indices: Tuple[int, int], q_lo: np.ndarray, q_hi: np.ndarray, dt: float, guidance_cfg: Any, use_torch_fk: bool=False) -> Dict[str, torch.Tensor]:
    contact = build_contact_channels(keypoints, fq, sk=sk, foot_indices=foot_indices, dt=float(dt), guidance_cfg=guidance_cfg, use_torch_fk=use_torch_fk)
    foot_l, foot_r = _foot_positions_torch(sk, foot_indices, fq, use_torch_fk=use_torch_fk)
    out: Dict[str, torch.Tensor] = {}
    for key in CONTACT_CHANNEL_KEYS:
        out[key] = contact[key]
    id_out, tau, qdot = build_id_channels(fq, sk=sk, foot_indices=foot_indices, dt=float(dt))
    out.update(id_out)
    out.update(build_limit_channels(fq, q_lo, q_hi))
    out.update(build_symmetry_channels(fq))
    out.update(build_clearance_channels(keypoints))
    out.update(build_derived_channels(tau=tau, qdot=qdot, contact=contact, fq=fq, q_lo=q_lo, q_hi=q_hi, foot_l=foot_l, foot_r=foot_r))
    return {k: out[k] for k in BIOMECH_COMPONENT_KEYS}