"""B3D pack/unpack for cached muscle activations ``[80, T]``."""

from __future__ import annotations

import numpy as np

# Rajagopal 2015 full-body muscle count (fixed for bundled skeleton).
MUSCLE_ACTIVATION_ROWS = 80


def pack_muscle_activations(activations: np.ndarray) -> np.ndarray:
    """``activations`` ``[T, M]`` → B3D matrix ``[M, T]`` float64."""
    arr = np.asarray(activations, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != MUSCLE_ACTIVATION_ROWS:
        raise ValueError(f"Expected activations [T, {MUSCLE_ACTIVATION_ROWS}], got {arr.shape}")
    return np.ascontiguousarray(arr.T)


def is_zero_placeholder_activations(
    activations: np.ndarray,
    *,
    atol: float = 1e-8,
) -> bool:
    """True when OpenSim labels are all-zero fallbacks (failed / skipped solve)."""
    act = np.asarray(activations)
    if act.size == 0:
        return True
    return bool(np.allclose(act, 0.0, atol=float(atol), equal_nan=True))


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
