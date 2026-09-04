"""Pose utilities shared by OpenSim IK. Nimblephysics joint IK removed."""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np

NUM_HML3D_JOINTS = 22
# Legacy Nimble Rajagopal IK mapping — removed; OpenSim uses opensim_ik.HML3D_MARKER_MAP.
HML3D_IK: Tuple[Tuple[str, int], ...] = ()


def clear_body_ik_cache() -> None:
    return None


def pose_is_invalid(q: np.ndarray, *, atol: float = 1e-06) -> bool:
    v = np.asarray(q, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return True
    return bool(np.allclose(v, 0.0, atol=atol) or np.linalg.norm(v) < atol)


def fill_invalid_pose_frames(poses_q: np.ndarray, *, atol: float = 1e-06) -> Tuple[np.ndarray, int]:
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


def fit_q(*_a: Any, **_k: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    raise RuntimeError('nimblephysics fit_q removed; use nimble.opensim_ik.fit_q_lai')
