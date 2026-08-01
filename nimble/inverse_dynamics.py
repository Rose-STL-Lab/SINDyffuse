from __future__ import annotations
from typing import Any, List, Sequence, Tuple
import numpy as np
import torch
from nimble.dof_groups import ID_GROUP_SPECS, ID_SYMMETRY_SPECS
from nimble.ops import finite_diff, safe_norm

def _zero_velocity_penalty(_: float) -> float:
    return 0.0

def _contact_body_nodes(sk: Any, foot_indices: Tuple[int, int]) -> List[Any]:
    left_idx, right_idx = foot_indices
    return [sk.getBodyNode(int(left_idx)), sk.getBodyNode(int(right_idx))]

def compute_joint_torques_numpy(q: np.ndarray, *, sk: Any, foot_indices: Tuple[int, int], dt: float, use_contact_id: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(q, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f'Expected q [T, ndof], got {arr.shape}')
    t_len, ndof = (int(arr.shape[0]), int(arr.shape[1]))
    if t_len < 2:
        z = np.zeros((t_len, ndof), dtype=np.float64)
        return (z, z)
    if use_contact_id and t_len >= 3:
        q_cols = arr.T
        contact_bodies = _contact_body_nodes(sk, foot_indices)
        try:
            result = sk.getMultipleContactInverseDynamicsOverTime(q_cols, contact_bodies, smoothingWeight=1.0, minTorqueWeight=1.0, velocityPenalty=_zero_velocity_penalty)
            tau_cols = np.asarray(result.jointTorques, dtype=np.float64)
            vel_cols = np.asarray(result.velocities, dtype=np.float64)
            if tau_cols.ndim == 2 and tau_cols.shape[0] == ndof:
                tau = tau_cols.T
                qdot = vel_cols.T if vel_cols.ndim == 2 else np.zeros_like(tau)
                if tau.shape[0] == t_len:
                    return (tau, qdot)
                if tau.shape[0] < t_len:
                    pad = t_len - tau.shape[0]
                    tau = np.pad(tau, ((0, pad), (0, 0)), mode='edge')
                    qdot = np.pad(qdot, ((0, pad), (0, 0)), mode='edge')
                    return (tau[:t_len], qdot[:t_len])
        except Exception:
            pass
    qdot = np.zeros_like(arr)
    qddot = np.zeros_like(arr)
    qdot[0] = (arr[1] - arr[0]) / float(dt)
    qdot[-1] = (arr[-1] - arr[-2]) / float(dt)
    if t_len > 2:
        qdot[1:-1] = (arr[2:] - arr[:-2]) / (2.0 * float(dt))
        qddot[1:-1] = (qdot[2:] - qdot[:-2]) / (2.0 * float(dt))
    qddot[0] = (qdot[1] - qdot[0]) / float(dt)
    qddot[-1] = (qdot[-1] - qdot[-2]) / float(dt)
    tau = np.zeros_like(arr)
    contact_bodies = _contact_body_nodes(sk, foot_indices)
    for t in range(t_len):
        sk.setPositions(arr[t].reshape(-1))
        sk.setVelocities(qdot[t].reshape(-1))
        acc = qddot[t].reshape(-1, 1)
        try:
            if use_contact_id:
                id_result = sk.getMultipleContactInverseDynamics(acc, contact_bodies)
                row = np.asarray(id_result.jointTorques, dtype=np.float64).reshape(-1)
            else:
                row = np.asarray(sk.getInverseDynamics(acc), dtype=np.float64).reshape(-1)
        except Exception:
            row = np.asarray(sk.getInverseDynamics(acc), dtype=np.float64).reshape(-1)
        if row.shape[0] == ndof:
            tau[t] = row
    return (tau, qdot)

class _InverseDynamicsNimble(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, q: torch.Tensor, sk: Any, foot_indices: Tuple[int, int], dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        q_np = q.detach().contiguous().double().cpu().numpy()
        tau_np, qdot_np = compute_joint_torques_numpy(q_np, sk=sk, foot_indices=foot_indices, dt=float(dt))
        ctx.sk = sk
        ctx.foot_indices = tuple((int(i) for i in foot_indices))
        ctx.dt = float(dt)
        ctx.save_for_backward(q)
        tau = torch.as_tensor(tau_np, dtype=q.dtype, device=q.device)
        qdot = torch.as_tensor(qdot_np, dtype=q.dtype, device=q.device)
        return (tau, qdot)

    @staticmethod
    def backward(ctx: Any, grad_tau: torch.Tensor, grad_qdot: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        q, = ctx.saved_tensors
        if grad_tau is None or not torch.any(grad_tau != 0):
            return (torch.zeros_like(q), None, None, None)
        eps = 0.0001
        q_np = q.detach().contiguous().double().cpu().numpy()
        base_tau, _ = compute_joint_torques_numpy(q_np, sk=ctx.sk, foot_indices=ctx.foot_indices, dt=ctx.dt)
        g_np = grad_tau.detach().contiguous().double().cpu().numpy()
        grad_q = np.zeros_like(q_np)
        for d in range(q_np.shape[1]):
            qp = q_np.copy()
            qp[:, d] += eps
            tau_p, _ = compute_joint_torques_numpy(qp, sk=ctx.sk, foot_indices=ctx.foot_indices, dt=ctx.dt)
            grad_q[:, d] = np.sum((tau_p - base_tau) * g_np, axis=1) / eps
        return (torch.as_tensor(grad_q, dtype=q.dtype, device=q.device), None, None, None)

def joint_torques_torch(q: torch.Tensor, *, sk: Any, foot_indices: Tuple[int, int], dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
    return _InverseDynamicsNimble.apply(q, sk, foot_indices, float(dt))

def id_group_norms(tau: torch.Tensor) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, sl in ID_GROUP_SPECS:
        chunk = tau[:, sl]
        out[name] = safe_norm(chunk, dim=1)
    return out

def id_symmetry_scalars(tau: torch.Tensor) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, sl_l, sl_r in ID_SYMMETRY_SPECS:
        diff = tau[:, sl_l] - tau[:, sl_r]
        out[name] = safe_norm(diff, dim=1)
    return out

def id_power_total(tau: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
    return torch.abs((tau * qdot).sum(dim=1, keepdim=True))