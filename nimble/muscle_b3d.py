"""B3D pack/unpack for cached muscle activations ``[80, T]``."""

from __future__ import annotations

import numpy as np

from nimble.muscle_activation import muscle_names as _muscle_names

MUSCLE_ACTIVATION_ROWS = len(_muscle_names())
MUSCLE_ACTIVATION_MASK_ROWS = 1


def pack_muscle_activations(activations: np.ndarray) -> np.ndarray:
    """``activations`` ``[T, M]`` → B3D matrix ``[M, T]`` float64."""
    arr = np.asarray(activations, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != MUSCLE_ACTIVATION_ROWS:
        raise ValueError(f"Expected activations [T, {MUSCLE_ACTIVATION_ROWS}], got {arr.shape}")
    return np.ascontiguousarray(arr.T)


def unpack_muscle_activations(matrix: np.ndarray) -> np.ndarray:
    """B3D ``[M, T]`` or ``[T, M]`` → ``[T, M]`` float32."""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == MUSCLE_ACTIVATION_ROWS:
        return arr.T
    if arr.ndim == 2 and arr.shape[1] == MUSCLE_ACTIVATION_ROWS:
        return arr
    raise ValueError(
        f"Expected muscle_activations layout with {MUSCLE_ACTIVATION_ROWS} channels, got {arr.shape}"
    )


def pack_muscle_activation_mask(label_valid: np.ndarray) -> np.ndarray:
    """``label_valid`` ``[T]`` bool/0-1 → B3D matrix ``[1, T]`` float64."""
    v = np.asarray(label_valid, dtype=np.float64).reshape(-1)
    if v.ndim != 1:
        raise ValueError(f"Expected label_valid [T], got {v.shape}")
    return np.ascontiguousarray(v.reshape(MUSCLE_ACTIVATION_MASK_ROWS, -1))


def unpack_muscle_activation_mask(matrix: np.ndarray) -> np.ndarray:
    """B3D ``[1, T]`` or ``[T]`` → ``[T]`` float32 in ``{0, 1}``."""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == MUSCLE_ACTIVATION_MASK_ROWS:
        return (arr.reshape(-1) > 0.5).astype(np.float32)
    if arr.ndim == 1:
        return (arr.reshape(-1) > 0.5).astype(np.float32)
    if arr.ndim == 2 and arr.shape[1] == MUSCLE_ACTIVATION_MASK_ROWS:
        return (arr[:, 0] > 0.5).astype(np.float32)
    raise ValueError(
        f"Expected muscle_activation_mask layout with {MUSCLE_ACTIVATION_MASK_ROWS} row(s), "
        f"got {arr.shape}"
    )
