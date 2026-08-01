from __future__ import annotations
import numpy as np
MUSCLE_ACTIVATION_ROWS = 80

def pack_muscle_activations(activations: np.ndarray) -> np.ndarray:
    arr = np.asarray(activations, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != MUSCLE_ACTIVATION_ROWS:
        raise ValueError(f'Expected activations [T, {MUSCLE_ACTIVATION_ROWS}], got {arr.shape}')
    return np.ascontiguousarray(arr.T)

def is_zero_placeholder_activations(activations: np.ndarray, *, atol: float=1e-08) -> bool:
    act = np.asarray(activations)
    if act.size == 0:
        return True
    return bool(np.allclose(act, 0.0, atol=float(atol), equal_nan=True))

def unpack_muscle_activations(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == MUSCLE_ACTIVATION_ROWS:
        return arr.T
    if arr.ndim == 2 and arr.shape[1] == MUSCLE_ACTIVATION_ROWS:
        return arr
    raise ValueError(f'Expected muscle_activations layout with {MUSCLE_ACTIVATION_ROWS} channels, got {arr.shape}')