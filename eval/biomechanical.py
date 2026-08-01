from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import torch
from nimble.contact import estimate_contact_from_feet
from nimble.ops import finite_diff
from nimble.physics import load_model
from nimble.rajagopal_kin import COM_KEYPOINT_INDICES, IDX_FOOT_L, IDX_FOOT_R, IDX_PELVIS, keypoints_numpy

def compute_fidelity(q: np.ndarray, q_ref: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64).reshape(1, -1)
    std = np.asarray(std, dtype=np.float64).reshape(1, -1)
    q_n = (q - mean) / np.clip(std, 1e-06, None)
    q_ref_n = (q_ref - mean) / np.clip(std, 1e-06, None)
    mse = np.mean((q_n - q_ref_n) ** 2)
    return float(np.sqrt(mse))

def compute_biomechanical_metrics(q: np.ndarray, *, fps: float=20.0, q_ref: Optional[np.ndarray]=None, mean: Optional[np.ndarray]=None, std: Optional[np.ndarray]=None, mass_kg: float=70.0, height_thresh_m: float=0.06, speed_thresh_mps: float=1.2) -> Dict[str, float]:
    q = np.asarray(q, dtype=np.float32)
    if q.ndim != 2:
        raise ValueError(f'Expected q [T, ndof], got {q.shape}')
    sk = load_model().skeleton
    kp = keypoints_numpy(sk, q)
    dt = 1.0 / float(fps)
    kp_t = torch.from_numpy(kp)
    pelvis = kp_t[:, IDX_PELVIS, :]
    root_v = finite_diff(pelvis, dt)
    root_a = finite_diff(root_v, dt)
    speed = float(torch.linalg.norm(root_v, dim=1).mean().item())
    root_acc = float(torch.linalg.norm(root_a, dim=1).mean().item())
    foot_l = kp_t[:, IDX_FOOT_L, :]
    foot_r = kp_t[:, IDX_FOOT_R, :]
    com_pos = kp_t[:, list(COM_KEYPOINT_INDICES), :].mean(dim=1)
    com_v = finite_diff(com_pos, dt)
    com_a = finite_diff(com_v, dt)
    contact = estimate_contact_from_feet(foot_l, foot_r, com_a, dt, mass_kg=float(mass_kg), height_thresh_m=float(height_thresh_m), speed_thresh_mps=float(speed_thresh_mps))
    contact_lr = contact['inferred_contact_active_lr']
    ground_y = torch.minimum(foot_l[:, 1].min(), foot_r[:, 1].min())
    left_h = foot_l[:, 1] - ground_y
    right_h = foot_r[:, 1] - ground_y
    foot_h = torch.minimum(left_h, right_h)
    penetration_cm = float(torch.clamp(-foot_h, min=0.0).mean().item()) * 100.0
    c = contact_lr.max(dim=1).values
    floating_cm = float((c * torch.clamp(foot_h, min=0.0)).mean().item()) * 100.0
    left_vel = finite_diff(foot_l, dt)
    right_vel = finite_diff(foot_r, dt)
    foot_speed = contact_lr[:, 0] * torch.linalg.norm(left_vel[:, [0, 2]], dim=1)
    foot_speed = foot_speed + contact_lr[:, 1] * torch.linalg.norm(right_vel[:, [0, 2]], dim=1)
    sliding_mps = float(foot_speed.mean().item())
    out: Dict[str, float] = {'speed_mps': speed, 'root_acc_mps2': root_acc, 'penetration_cm': penetration_cm, 'floating_cm': floating_cm, 'sliding_mps': sliding_mps}
    if q_ref is not None and mean is not None and (std is not None):
        out['fidelity'] = compute_fidelity(q, q_ref, mean, std)
    return out