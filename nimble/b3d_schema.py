"""B3D ``customValues`` names and layout for SINDyffuse preprocessed motions."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np

from nimble.channels import BIOMECH_COMPONENT_KEYS
from nimble.muscle_b3d import (
    MUSCLE_ACTIVATION_ROWS,
    pack_muscle_activations,
    unpack_muscle_activations,
)
from nimble.muscle_activation import muscle_names, opensim_quiet
from nimble.physics import load_model
from sindy.features import features_from_q

GUIDANCE_FEATURES = "guidance_features"
SINDY_FEATURES = "sindy_features"
MUSCLE_ACTIVATIONS = "muscle_activations"

B3D_CUSTOM_VALUE_NAMES: Tuple[str, ...] = (
    GUIDANCE_FEATURES,
    SINDY_FEATURES,
    MUSCLE_ACTIVATIONS,
)

GUIDANCE_FEATURE_ROWS = len(BIOMECH_COMPONENT_KEYS)


@lru_cache(maxsize=1)
def _sindy_layout() -> Tuple[int, int, int, Tuple[str, ...], Tuple[str, ...]]:
    with opensim_quiet("Off"):
        sk = load_model().skeleton
        ndof = int(sk.getNumDofs())
        _, _, u_names, c_names = features_from_q(
            np.zeros((2, ndof), dtype=np.float64), sk, fps=20.0
        )
    u_rows = len(u_names)
    c_rows = len(c_names)
    return u_rows, c_rows, u_rows + c_rows, tuple(u_names), tuple(c_names)


def _sindy_u_rows() -> int:
    return _sindy_layout()[0]


def _sindy_c_rows() -> int:
    return _sindy_layout()[1]


def _sindy_feature_rows() -> int:
    return _sindy_layout()[2]


def _u_feature_names() -> Tuple[str, ...]:
    return _sindy_layout()[3]


def _c_feature_names() -> Tuple[str, ...]:
    return _sindy_layout()[4]


@lru_cache(maxsize=1)
def _muscle_names_quiet() -> Tuple[str, ...]:
    with opensim_quiet("Off"):
        return muscle_names()


# Backward-compatible module constants (lazy on first use via PEP 562).
def __getattr__(name: str):
    if name == "SINDY_U_ROWS":
        return _sindy_u_rows()
    if name == "SINDY_C_ROWS":
        return _sindy_c_rows()
    if name == "SINDY_FEATURE_ROWS":
        return _sindy_feature_rows()
    if name == "U_FEATURE_NAMES":
        return _u_feature_names()
    if name == "C_FEATURE_NAMES":
        return _c_feature_names()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def clear_b3d_schema_caches() -> None:
    """Drop layout caches that retain skeleton / OpenSim state."""
    _sindy_layout.cache_clear()
    _muscle_names_quiet.cache_clear()


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
    u_rows = _sindy_u_rows()
    c_rows = _sindy_c_rows()
    u_arr = np.asarray(u, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64)
    if u_arr.ndim != 2 or u_arr.shape[1] != u_rows:
        raise ValueError(f"Expected u [T, {u_rows}], got {u_arr.shape}")
    if c_arr.ndim != 2 or c_arr.shape[1] != c_rows:
        raise ValueError(f"Expected c [T, {c_rows}], got {c_arr.shape}")
    if u_arr.shape[0] != c_arr.shape[0]:
        raise ValueError(f"u/c length mismatch: {u_arr.shape[0]} vs {c_arr.shape[0]}")
    return np.ascontiguousarray(np.vstack([u_arr.T, c_arr.T]))


def unpack_sindy_features(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """B3D ``[U+C, T]`` or ``[T, U+C]`` → ``u``, ``c``."""
    u_rows = _sindy_u_rows()
    feature_rows = _sindy_feature_rows()
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == feature_rows:
        return arr[:u_rows, :].T, arr[u_rows:, :].T
    if arr.ndim == 2 and arr.shape[1] >= feature_rows:
        arr = arr[:, :feature_rows]
        return arr[:, :u_rows], arr[:, u_rows:]
    raise ValueError(f"Expected sindy_features with {feature_rows} rows, got {arr.shape}")


def metadata_custom_values_block() -> dict:
    return {
        GUIDANCE_FEATURES: {
            "rows": GUIDANCE_FEATURE_ROWS,
            "channel_order": list(BIOMECH_COMPONENT_KEYS),
        },
        SINDY_FEATURES: {
            "rows": _sindy_feature_rows(),
            "u_rows": _sindy_u_rows(),
            "c_rows": _sindy_c_rows(),
            "u_names": list(_u_feature_names()),
            "c_names": list(_c_feature_names()),
        },
        MUSCLE_ACTIVATIONS: {
            "rows": MUSCLE_ACTIVATION_ROWS,
            "muscle_names": list(_muscle_names_quiet()),
        },
    }
