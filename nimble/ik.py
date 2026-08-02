from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
NUM_HML3D_JOINTS = 22
from nimble.skeleton_registry import SKELETONS
HML3D_IK: Tuple[Tuple[str, int], ...] = SKELETONS['rajagopal'].ik_mapping

@dataclass
class _JointIkCache:
    joints: List[Any]
    joint_names: Tuple[str, ...]
    hml_indices: Tuple[int, ...]
_JOINT_IK_CACHE: Dict[Tuple[int, Tuple[Tuple[str, int], ...]], _JointIkCache] = {}

def clear_body_ik_cache() -> None:
    _JOINT_IK_CACHE.clear()

def pose_is_invalid(q: np.ndarray, *, atol: float=1e-06) -> bool:
    v = np.asarray(q, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return True
    return bool(np.allclose(v, 0.0, atol=atol) or np.linalg.norm(v) < atol)

def joint_fit_squared_loss(skeleton: Any, joints: List[Any], target_positions: np.ndarray) -> float:
    flat = np.asarray(skeleton.getJointWorldPositions(joints), dtype=np.float64).reshape(-1)
    target = np.asarray(target_positions, dtype=np.float64).reshape(-1)
    diff = flat - target
    return float(np.dot(diff, diff))

def fill_invalid_pose_frames(poses_q: np.ndarray, *, atol: float=1e-06) -> Tuple[np.ndarray, int]:
    out = np.asarray(poses_q, dtype=np.float64).copy()
    if out.ndim != 2:
        raise ValueError(f'Expected poses_q [num_dofs, T], got {out.shape}')
    t_frames = int(out.shape[1])
    filled = 0
    last_valid: np.ndarray | None = None
    for t in range(t_frames):
        if pose_is_invalid(out[:, t], atol=atol):
            if last_valid is not None:
                out[:, t] = last_valid
                filled += 1
        else:
            last_valid = out[:, t].copy()
    last_valid = None
    for t in range(t_frames - 1, -1, -1):
        if pose_is_invalid(out[:, t], atol=atol):
            if last_valid is not None:
                out[:, t] = last_valid
                filled += 1
        else:
            last_valid = out[:, t].copy()
    return (out, filled)

def _get_joint_ik_cache(skeleton: Any, *, ik_mapping: Tuple[Tuple[str, int], ...] | None=None) -> _JointIkCache:
    mapping = tuple(ik_mapping) if ik_mapping is not None else HML3D_IK
    key = (id(skeleton), mapping)
    if key not in _JOINT_IK_CACHE:
        joints: List[Any] = []
        names: List[str] = []
        hml_idx: List[int] = []
        missing: List[str] = []
        for joint_name, jidx in mapping:
            try:
                joints.append(skeleton.getJoint(str(joint_name)))
                names.append(str(joint_name))
                hml_idx.append(int(jidx))
            except Exception:
                missing.append(str(joint_name))
                continue
        if not joints:
            raise RuntimeError(f'Could not resolve any joints from IK mapping {mapping!r}. Missing: {missing}')
        if missing:
            import sys
            print(f'WARNING: IK mapping missing joints (skipped): {missing}', file=sys.stderr, flush=True)
        _JOINT_IK_CACHE[key] = _JointIkCache(joints=joints, joint_names=tuple(names), hml_indices=tuple(hml_idx))
    return _JOINT_IK_CACHE[key]

def _run_ik_pass(poses: np.ndarray, skeleton: Any, cache: _JointIkCache, *, scale_bodies: bool, line_search: bool, log_output: bool, frame_order: Optional[List[int]]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    t_frames = int(poses.shape[0])
    ndof = int(skeleton.getNumDofs())
    poses_q = np.zeros((ndof, t_frames), dtype=np.float64)
    fk_loss = np.full(t_frames, np.nan, dtype=np.float64)
    solver_err = np.full(t_frames, np.nan, dtype=np.float64)
    order: List[int] = list(range(t_frames)) if frame_order is None else [int(t) for t in frame_order]
    success = 0
    held_frames = 0
    last_q: np.ndarray | None = None
    joints_list = list(cache.joints)
    n_targets = len(cache.hml_indices)
    for t in order:
        target_xyz = np.array([poses[t, int(j), :] for j in cache.hml_indices], dtype=np.float64)
        target_joints_array = np.ascontiguousarray(target_xyz.reshape((-1, 1)))
        if last_q is not None:
            skeleton.setPositions(last_q)
        try:
            err = skeleton.fitJointsToWorldPositions(cache.joints, target_joints_array, scaleBodies=bool(scale_bodies), logOutput=bool(log_output), lineSearch=bool(line_search))
            err_f = float(err)
            q_frame = np.asarray(skeleton.getPositions(), dtype=np.float64).reshape(-1)
            if pose_is_invalid(q_frame) and last_q is not None:
                q_frame = last_q.copy()
                held_frames += 1
            else:
                if np.isfinite(err_f):
                    solver_err[t] = err_f
                flat = np.asarray(skeleton.getJointWorldPositions(joints_list), dtype=np.float64).reshape(n_targets, 3)
                diff = flat - target_xyz
                fk_loss[t] = float(np.sum(diff * diff))
                success += 1
            poses_q[:, t] = q_frame
            last_q = q_frame.copy()
        except Exception:
            if last_q is not None:
                poses_q[:, t] = last_q
                held_frames += 1
    return (poses_q, fk_loss, solver_err, success, held_frames)

def _scale_bones_to_first_frame(skeleton: Any, poses: np.ndarray, cache: _JointIkCache, *, line_search: bool, log_output: bool) -> None:
    if poses.shape[0] == 0:
        return
    target_arr = np.ascontiguousarray(np.array([poses[0, int(j), :] for j in cache.hml_indices], dtype=np.float64).reshape((-1, 1)))
    try:
        skeleton.fitJointsToWorldPositions(cache.joints, target_arr, scaleBodies=True, logOutput=bool(log_output), lineSearch=bool(line_search))
    except Exception:
        pass

def _aggregate_err_stats(arr: np.ndarray, prefix: str) -> Dict[str, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {f'mean_{prefix}': 0.0, f'median_{prefix}': 0.0, f'std_{prefix}': 0.0, f'min_{prefix}': 0.0, f'max_{prefix}': 0.0}
    return {f'mean_{prefix}': float(np.mean(finite)), f'median_{prefix}': float(np.median(finite)), f'std_{prefix}': float(np.std(finite)), f'min_{prefix}': float(np.min(finite)), f'max_{prefix}': float(np.max(finite))}

def fit_q(hml3d_positions: np.ndarray, skeleton: Any, *, scale_bodies: bool=False, line_search: bool=True, log_output: bool=False, ik_mapping: Tuple[Tuple[str, int], ...] | None=None, bidirectional: bool=True, scale_first_frame: bool=True) -> Tuple[np.ndarray, Dict[str, Any]]:
    poses = np.asarray(hml3d_positions, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (NUM_HML3D_JOINTS, 3):
        raise ValueError(f'Expected hml3d_positions [T,22,3], got {poses.shape}')
    t_frames = int(poses.shape[0])
    cache = _get_joint_ik_cache(skeleton, ik_mapping=ik_mapping)
    if scale_first_frame:
        _scale_bones_to_first_frame(skeleton, poses, cache, line_search=line_search, log_output=log_output)
    fwd_q, fwd_fk, fwd_solver, _fwd_success, fwd_held = _run_ik_pass(poses, skeleton, cache, scale_bodies=scale_bodies, line_search=line_search, log_output=log_output, frame_order=list(range(t_frames)))
    chose_bwd = 0
    if bidirectional and t_frames > 1:
        bwd_q, bwd_fk, bwd_solver, _, _ = _run_ik_pass(poses, skeleton, cache, scale_bodies=scale_bodies, line_search=line_search, log_output=log_output, frame_order=list(range(t_frames - 1, -1, -1)))
        final_q = fwd_q.copy()
        final_fk = fwd_fk.copy()
        final_solver = fwd_solver.copy()
        for t in range(t_frames):
            f_l = fwd_fk[t]
            b_l = bwd_fk[t]
            f_ok = np.isfinite(f_l)
            b_ok = np.isfinite(b_l)
            if b_ok and (not f_ok or b_l < f_l):
                final_q[:, t] = bwd_q[:, t]
                final_fk[t] = b_l
                final_solver[t] = bwd_solver[t]
                chose_bwd += 1
    else:
        final_q = fwd_q
        final_fk = fwd_fk
        final_solver = fwd_solver
    poses_q, filled_frames = fill_invalid_pose_frames(final_q)
    held_frames = fwd_held + filled_frames
    finite_solver = final_solver[np.isfinite(final_solver)]
    success = int(finite_solver.size)
    stats: Dict[str, Any] = {'total_frames': float(t_frames), 'success_count': float(success), 'failed_frames': float(t_frames - success), 'success_ratio': float(success / max(t_frames, 1)), 'held_frames': float(held_frames), 'filled_invalid_frames': float(filled_frames), 'bidirectional': float(bool(bidirectional)), 'bidi_chose_backward': float(chose_bwd), 'per_frame_loss': final_solver.tolist(), 'per_frame_fk_loss': final_fk.tolist()}
    solver_stats = _aggregate_err_stats(final_solver, 'ik_error')
    stats.update(solver_stats)
    stats.update({k.replace('ik_error', 'fit_joints_loss'): v for k, v in solver_stats.items()})
    stats.update(_aggregate_err_stats(final_fk, 'fk_loss'))
    return (poses_q, stats)