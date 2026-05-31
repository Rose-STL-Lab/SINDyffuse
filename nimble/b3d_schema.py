"""B3D ``customValues`` names and layout for SINDyffuse preprocessed motions."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from nimble.channels import BIOMECH_COMPONENT_KEYS
from nimble.physics import load_model
from sindy.features import features_from_q

GUIDANCE_FEATURES = "guidance_features"
SINDY_FEATURES = "sindy_features"
MUSCLE_ACTIVATIONS = "muscle_activations"

# Rajagopal 2015 full-body muscle count (fixed).
MUSCLE_ACTIVATION_ROWS = 80

B3D_CUSTOM_VALUE_NAMES: Tuple[str, ...] = (
    GUIDANCE_FEATURES,
    SINDY_FEATURES,
    MUSCLE_ACTIVATIONS,
)

GUIDANCE_FEATURE_ROWS = len(BIOMECH_COMPONENT_KEYS)

_sk = load_model().skeleton
_ndof = int(_sk.getNumDofs())
_, _, _U_NAMES, _C_NAMES = features_from_q(
    np.zeros((2, _ndof), dtype=np.float64), _sk, fps=20.0
)
SINDY_U_ROWS = len(_U_NAMES)
SINDY_C_ROWS = len(_C_NAMES)
SINDY_FEATURE_ROWS = SINDY_U_ROWS + SINDY_C_ROWS

U_FEATURE_NAMES: Tuple[str, ...] = tuple(_U_NAMES)
C_FEATURE_NAMES: Tuple[str, ...] = tuple(_C_NAMES)


def pack_guidance_features(bio: np.ndarray) -> np.ndarray:
    """``bio`` ``[T, C]`` → B3D matrix ``[C, T]`` float64."""
    arr = np.asarray(bio, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != GUIDANCE_FEATURE_ROWS:
        raise ValueError(f"Expected bio [T, {GUIDANCE_FEATURE_ROWS}], got {arr.shape}")
    return np.ascontiguousarray(arr.T)


def unpack_guidance_features(matrix: np.ndarray) -> np.ndarray:
    """B3D ``[C, T]`` or frame stack ``[T, C]`` → ``[T, C]`` float32."""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == GUIDANCE_FEATURE_ROWS:
        return arr.T
    if arr.ndim == 2 and arr.shape[1] == GUIDANCE_FEATURE_ROWS:
        return arr
    raise ValueError(f"Expected guidance layout with {GUIDANCE_FEATURE_ROWS} channels, got {arr.shape}")


def pack_sindy_features(u: np.ndarray, c: np.ndarray) -> np.ndarray:
    """``u`` ``[T, U]``, ``c`` ``[T, C]`` → B3D matrix ``[U+C, T]`` float64."""
    u_arr = np.asarray(u, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64)
    if u_arr.ndim != 2 or u_arr.shape[1] != SINDY_U_ROWS:
        raise ValueError(f"Expected u [T, {SINDY_U_ROWS}], got {u_arr.shape}")
    if c_arr.ndim != 2 or c_arr.shape[1] != SINDY_C_ROWS:
        raise ValueError(f"Expected c [T, {SINDY_C_ROWS}], got {c_arr.shape}")
    if u_arr.shape[0] != c_arr.shape[0]:
        raise ValueError(f"u/c length mismatch: {u_arr.shape[0]} vs {c_arr.shape[0]}")
    return np.ascontiguousarray(np.vstack([u_arr.T, c_arr.T]))


def unpack_sindy_features(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """B3D ``[U+C, T]`` or ``[T, U+C]`` → ``u``, ``c``."""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == SINDY_FEATURE_ROWS:
        return arr[:SINDY_U_ROWS, :].T, arr[SINDY_U_ROWS:, :].T
    if arr.ndim == 2 and arr.shape[1] >= SINDY_FEATURE_ROWS:
        arr = arr[:, :SINDY_FEATURE_ROWS]
        return arr[:, :SINDY_U_ROWS], arr[:, SINDY_U_ROWS:]
    raise ValueError(f"Expected sindy_features with {SINDY_FEATURE_ROWS} rows, got {arr.shape}")


def pack_muscle_activations(activations: np.ndarray) -> np.ndarray:
    """``activations`` ``[T, M]`` → B3D matrix ``[M, T]`` float64."""
    from surrogate.b3d_activation import pack_muscle_activations as _pack

    return _pack(activations)


def unpack_muscle_activations(matrix: np.ndarray) -> np.ndarray:
    """B3D ``[M, T]`` or ``[T, M]`` → ``[T, M]`` float32."""
    from surrogate.b3d_activation import unpack_muscle_activations as _unpack

    return _unpack(matrix)


def metadata_custom_values_block() -> dict:
    from surrogate.opensim_activation import muscle_names

    return {
        GUIDANCE_FEATURES: {
            "rows": GUIDANCE_FEATURE_ROWS,
            "channel_order": list(BIOMECH_COMPONENT_KEYS),
        },
        SINDY_FEATURES: {
            "rows": SINDY_FEATURE_ROWS,
            "u_rows": SINDY_U_ROWS,
            "c_rows": SINDY_C_ROWS,
            "u_names": list(U_FEATURE_NAMES),
            "c_names": list(C_FEATURE_NAMES),
        },
        MUSCLE_ACTIVATIONS: {
            "rows": MUSCLE_ACTIVATION_ROWS,
            "muscle_names": list(muscle_names()),
        },
    }
