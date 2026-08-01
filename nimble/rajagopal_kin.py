from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
FOOT_BODY_NAMES: Tuple[str, ...] = ('calcn_l', 'calcn_r')

def _foot_body_indices(sk: Any) -> Tuple[int, int]:
    return tuple((int(sk.getBodyNode(str(n)).getIndexInSkeleton()) for n in FOOT_BODY_NAMES))

def _load_model():
    from nimble.physics import load_model
    return load_model()
KEYPOINT_JOINT_NAMES: Tuple[str, ...] = ('ground_pelvis', 'hip_l', 'hip_r', 'back', 'mtp_l', 'mtp_r')
NUM_KEYPOINTS = len(KEYPOINT_JOINT_NAMES)
IDX_PELVIS = 0
IDX_HIP_L = 1
IDX_HIP_R = 2
IDX_BACK = 3
IDX_FOOT_L = 4
IDX_FOOT_R = 5
COM_KEYPOINT_INDICES: Tuple[int, ...] = (IDX_PELVIS, IDX_HIP_L, IDX_HIP_R, IDX_BACK)
_KIN_CACHE: Dict[int, Tuple[Any, Tuple[int, ...], Tuple[int, int]]] = {}

def _resolve_keypoint_cache(sk: Any) -> Tuple[Any, Tuple[int, ...], Tuple[int, int]]:
    key = id(sk)
    if key not in _KIN_CACHE:
        joint_idx: List[int] = []
        for name in KEYPOINT_JOINT_NAMES:
            joint_idx.append(int(sk.getJoint(str(name)).getJointIndexInSkeleton()))
        _KIN_CACHE[key] = (sk, tuple(joint_idx), _foot_body_indices(sk))
    return _KIN_CACHE[key]

def clear_rajagopal_kin_cache() -> None:
    _KIN_CACHE.clear()

def foot_body_indices(sk: Any | None=None) -> Tuple[int, int]:
    if sk is None:
        sk = _load_model().skeleton
    return _resolve_keypoint_cache(sk)[2]

def keypoints_numpy(sk: Any, q: np.ndarray) -> np.ndarray:
    q_arr = np.asarray(q, dtype=np.float64)
    if q_arr.ndim != 2:
        raise ValueError(f'Expected q [T, ndof], got {q_arr.shape}')
    _, joint_indices, _ = _resolve_keypoint_cache(sk)
    t_frames = int(q_arr.shape[0])
    out = np.zeros((t_frames, NUM_KEYPOINTS, 3), dtype=np.float64)
    joint_cache: Dict[int, Any] = {}
    for t in range(t_frames):
        sk.setPositions(q_arr[t].reshape(-1))
        sk.computeForwardKinematics()
        for k, j_idx in enumerate(joint_indices):
            if j_idx not in joint_cache:
                joint_cache[j_idx] = sk.getJoint(int(j_idx))
            pos = sk.getJointWorldPositions([joint_cache[j_idx]])
            out[t, k, :] = np.asarray(pos, dtype=np.float64).reshape(3)
    return out.astype(np.float32)

def _joint_jacobian_np(sk: Any, joint: Any, q_np: np.ndarray) -> np.ndarray:
    sk.setPositions(q_np.reshape(-1))
    jac = sk.getJointWorldPositionsJacobianWrtJointPositions([joint])
    return np.asarray(jac, dtype=np.float64)

def _body_jacobian_np(sk: Any, body_idx: int, q_np: np.ndarray) -> np.ndarray:
    sk.setPositions(q_np.reshape(-1))
    body = sk.getBodyNode(int(body_idx))
    return np.asarray(sk.getLinearJacobian(body, np.zeros(3, dtype=np.float64)), dtype=np.float64)

class _RajagopalKeypointsNimble(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, q: torch.Tensor, joint_indices: Tuple[int, ...]) -> torch.Tensor:
        sk = _load_model().skeleton
        q_np = q.detach().contiguous().double().cpu().numpy()
        t_frames = int(q_np.shape[0])
        out = np.zeros((t_frames, NUM_KEYPOINTS, 3), dtype=np.float64)
        joint_cache: Dict[int, Any] = {}
        for t in range(t_frames):
            sk.setPositions(q_np[t])
            sk.computeForwardKinematics()
            for k, j_idx in enumerate(joint_indices):
                if j_idx not in joint_cache:
                    joint_cache[j_idx] = sk.getJoint(int(j_idx))
                out[t, k, :] = np.asarray(sk.getJointWorldPositions([joint_cache[j_idx]]), dtype=np.float64).reshape(3)
        ctx.joint_indices = tuple((int(i) for i in joint_indices))
        ctx.save_for_backward(q)
        return torch.as_tensor(out, dtype=q.dtype, device=q.device)

    @staticmethod
    def backward(ctx: Any, grad_kp: torch.Tensor) -> Tuple[torch.Tensor, None]:
        sk = _load_model().skeleton
        q, = ctx.saved_tensors
        g_np = grad_kp.detach().contiguous().double().cpu().numpy()
        grad_q = np.zeros_like(q.detach().contiguous().double().cpu().numpy())
        joint_cache: Dict[int, Any] = {}
        for t in range(int(grad_q.shape[0])):
            q_row = q[t].detach().contiguous().double().cpu().numpy().reshape(-1)
            sk.setPositions(q_row)
            for k, j_idx in enumerate(ctx.joint_indices):
                g = g_np[t, k, :]
                if np.allclose(g, 0.0):
                    continue
                if j_idx not in joint_cache:
                    joint_cache[j_idx] = sk.getJoint(int(j_idx))
                j = _joint_jacobian_np(sk, joint_cache[j_idx], q_row)
                grad_q[t] += j.T @ g
        return (torch.as_tensor(grad_q, dtype=q.dtype, device=q.device), None)

class _BodyOriginNimble(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, body_idx: int, q: torch.Tensor) -> torch.Tensor:
        sk = _load_model().skeleton
        q_np = q.detach().contiguous().double().cpu().numpy().reshape(-1)
        sk.setPositions(q_np)
        sk.computeForwardKinematics()
        tr = sk.getBodyNode(int(body_idx)).getWorldTransform()
        if hasattr(tr, 'translation'):
            pos = np.asarray(tr.translation(), dtype=np.float64).reshape(3)
        else:
            pos = np.asarray(tr, dtype=np.float64).reshape(3)
        ctx.body_idx = int(body_idx)
        ctx.save_for_backward(q)
        return torch.as_tensor(pos, dtype=q.dtype, device=q.device)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[None, torch.Tensor]:
        sk = _load_model().skeleton
        q, = ctx.saved_tensors
        g = grad_out.detach().contiguous().double().cpu().numpy().reshape(3)
        q_np = q.detach().contiguous().double().cpu().numpy().reshape(-1)
        j = _body_jacobian_np(sk, int(ctx.body_idx), q_np)
        grad_q = j.T @ g
        return (None, torch.as_tensor(grad_q, dtype=q.dtype, device=q.device))

def keypoints_torch(q: torch.Tensor) -> torch.Tensor:
    if q.ndim != 2:
        raise ValueError(f'Expected q [T, ndof], got {tuple(q.shape)}')
    _, joint_indices, _ = _resolve_keypoint_cache(_load_model().skeleton)
    return _RajagopalKeypointsNimble.apply(q, joint_indices)

def body_origins(q: torch.Tensor, body_indices: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    left_idx, right_idx = body_indices
    rows_l = [_BodyOriginNimble.apply(int(left_idx), q[t]) for t in range(int(q.shape[0]))]
    rows_r = [_BodyOriginNimble.apply(int(right_idx), q[t]) for t in range(int(q.shape[0]))]
    return (torch.stack(rows_l, dim=0), torch.stack(rows_r, dim=0))