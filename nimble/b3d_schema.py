"""Legacy B3D custom-value schema — kept for name constants only."""
from __future__ import annotations
from typing import Any
import numpy as np

GUIDANCE_FEATURES = 'guidance_features'
MUSCLE_ACTIVATIONS = 'muscle_activations'
MUSCLE_ACTIVATION_MASK = 'muscle_activation_mask'
SIM_GRF = 'sim_grf'
SINDY_FEATURES = 'sindy_features'


def unpack_activation_mask(raw: np.ndarray, t_len: int) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.size >= t_len:
        return arr[:t_len]
    out = np.zeros((t_len,), dtype=np.float32)
    out[: arr.size] = arr
    return out


def unpack_muscle_activations(raw: np.ndarray, t_len: int, n_muscles: int) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    need = t_len * n_muscles
    if arr.size >= need:
        return arr[:need].reshape(t_len, n_muscles)
    out = np.full((t_len, n_muscles), np.nan, dtype=np.float32)
    flat = min(arr.size, need)
    out.reshape(-1)[:flat] = arr[:flat]
    return out


def unpack_sim_grf(raw: np.ndarray, t_len: int, n_ch: int = 18) -> np.ndarray:
    return unpack_muscle_activations(raw, t_len, n_ch)


def unpack_guidance_features(raw: np.ndarray, t_len: int, n_feat: int) -> np.ndarray:
    return unpack_muscle_activations(raw, t_len, n_feat)


def unpack_sindy_features(raw: np.ndarray, t_len: int, n_feat: int) -> np.ndarray:
    return unpack_muscle_activations(raw, t_len, n_feat)


def require_skeleton_dofs(*_a: Any, **_k: Any) -> None:
    raise RuntimeError('B3D skeleton DOF checks removed; use LAI_CACHE_DOF_NAMES / LAI_NUM_DOFS')
