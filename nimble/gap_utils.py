from __future__ import annotations
from typing import Any
import numpy as np
from nimble.b3d_io import read_muscle_activation_mask_frames
from nimble.muscle_b3d import is_zero_placeholder_activations

def read_activation_validity_mask_frames(subj: Any, trial: int, start_frame: int, num_frames: int) -> np.ndarray:
    return read_muscle_activation_mask_frames(subj, trial, start_frame, num_frames)

def is_nan_placeholder_activations(act: np.ndarray, *, nan_frac_max: float=0.01) -> bool:
    arr = np.asarray(act, dtype=np.float64)
    if arr.size == 0:
        return True
    if is_zero_placeholder_activations(arr):
        return True
    nan_frac = float(np.mean(~np.isfinite(arr)))
    return nan_frac > float(nan_frac_max)

def window_is_valid(mask: np.ndarray, act: np.ndarray, *, min_valid_fraction: float=0.95) -> bool:
    m = np.asarray(mask, dtype=np.float64).reshape(-1)
    if m.size == 0:
        return False
    valid_frac = float(np.mean(m > 0.5))
    if valid_frac < float(min_valid_fraction):
        return False
    a = np.asarray(act, dtype=np.float64)
    if a.ndim == 2 and a.shape[0] == m.size:
        finite_rows = np.isfinite(a).all(axis=1)
        if float(np.mean(finite_rows & (m > 0.5))) < float(min_valid_fraction):
            return False
    return True

def motion_has_valid_activations(subj: Any, trial: int, tlen: int, *, zero_atol: float=1e-08, nan_frac_max: float=0.01) -> bool:
    from nimble.b3d_io import read_muscle_activations_frames
    act = read_muscle_activations_frames(subj, trial, 0, tlen)
    if is_nan_placeholder_activations(act, nan_frac_max=nan_frac_max):
        return False
    try:
        mask = read_activation_validity_mask_frames(subj, trial, 0, tlen)
    except Exception:
        return not is_zero_placeholder_activations(act, nan_frac_max=nan_frac_max)
    return float(np.sum(mask > 0.5)) > 0.0
