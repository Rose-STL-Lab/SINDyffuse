from __future__ import annotations
from typing import Any
import numpy as np
from nimble.muscle_b3d import is_zero_placeholder_activations


def read_activation_validity_mask_frames(data_or_subj: Any, trial: int, start_frame: int, num_frames: int) -> np.ndarray:
    """Accepts a lai_cache dict (preferred) or raises for legacy SubjectOnDisk."""
    del trial
    if isinstance(data_or_subj, dict) and 'muscle_activation_mask' in data_or_subj:
        mask = np.asarray(data_or_subj['muscle_activation_mask'], dtype=np.float32).reshape(-1)
        st = int(start_frame)
        return mask[st : st + int(num_frames)]
    raise TypeError('read_activation_validity_mask_frames expects a lai_cache motion dict')


def is_nan_placeholder_activations(act: np.ndarray, *, nan_frac_max: float = 0.01) -> bool:
    arr = np.asarray(act, dtype=np.float64)
    if arr.size == 0:
        return True
    if is_zero_placeholder_activations(arr):
        return True
    nan_frac = float(np.mean(~np.isfinite(arr)))
    return nan_frac > float(nan_frac_max)


def window_is_valid(mask: np.ndarray, act: np.ndarray, *, min_valid_fraction: float = 0.95) -> bool:
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


def motion_has_valid_activations(data: Any, trial: int = 0, tlen: int | None = None, *, zero_atol: float = 1e-08, nan_frac_max: float = 0.01) -> bool:
    del trial, zero_atol
    if not isinstance(data, dict):
        raise TypeError('motion_has_valid_activations expects a lai_cache motion dict')
    act = np.asarray(data['muscle_activations'], dtype=np.float64)
    if tlen is not None:
        act = act[: int(tlen)]
    if is_nan_placeholder_activations(act, nan_frac_max=nan_frac_max):
        return False
    mask = np.asarray(data['muscle_activation_mask'], dtype=np.float64).reshape(-1)
    if tlen is not None:
        mask = mask[: int(tlen)]
    return float(np.sum(mask > 0.5)) > 0.0
