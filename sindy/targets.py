"""L_bio + muscle activation targets from MinT q for SINDy training."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence, Tuple

import numpy as np

from common.biomech import BIOMECH_COMPONENT_KEYS
from common.skeleton_config import (
    muscle_channel_names as _muscle_channel_names,
    n_bio_targets,
    n_muscle_targets,
    n_sindy_targets,
)
from osim.physics import BioPhysicsConfig, bio_matrix_mint

N_BIO_TARGETS = n_bio_targets()
N_MUSCLE_TARGETS = n_muscle_targets()
N_SINDY_TARGETS = n_sindy_targets()


@lru_cache(maxsize=1)
def muscle_channel_names() -> Tuple[str, ...]:
    """Ordered muscle names for the MinT skeleton."""
    return _muscle_channel_names()


@lru_cache(maxsize=1)
def sindy_target_keys() -> Tuple[str, ...]:
    """All SINDy target channel names: bio + muscle."""
    return tuple(BIOMECH_COMPONENT_KEYS) + muscle_channel_names()


def default_physics_cfg(
    *,
    fps: float = 20.0,
    max_frames: int | None = None,
) -> BioPhysicsConfig:
    """Physics config for L_bio extraction."""
    t_max = 64 if max_frames is None else int(max_frames)
    return BioPhysicsConfig(max_physics_frames=t_max)


def bio_matrix(
    q: np.ndarray,
    *,
    fps: float,
    guidance_cfg: BioPhysicsConfig | None = None,
) -> np.ndarray:
    """``q`` ``[T, ndof]`` → L_bio channels ``[T, C]``."""
    return bio_matrix_mint(q, fps=fps, guidance_cfg=guidance_cfg)


def targets_for_theta(bio: np.ndarray) -> np.ndarray:
    """Align per-frame rows with SINDy theta length ``T-1`` (drop last frame)."""
    if bio.ndim != 2:
        raise ValueError(f"Expected targets [T,C], got {bio.shape}")
    if bio.shape[0] < 2:
        return bio[:0]
    return bio[:-1, :].astype(np.float32)


def build_sindy_targets(bio: np.ndarray, activations: np.ndarray) -> np.ndarray:
    """Concat bio and muscle activations → ``[T-1, n_targets]``."""
    bio_arr = np.asarray(bio, dtype=np.float32)
    act_arr = np.asarray(activations, dtype=np.float32)
    if bio_arr.ndim != 2 or bio_arr.shape[1] != N_BIO_TARGETS:
        raise ValueError(f"Expected bio [T, {N_BIO_TARGETS}], got {bio_arr.shape}")
    if act_arr.ndim != 2 or act_arr.shape[1] != N_MUSCLE_TARGETS:
        raise ValueError(f"Expected activations [T, {N_MUSCLE_TARGETS}], got {act_arr.shape}")
    if bio_arr.shape[0] != act_arr.shape[0]:
        raise ValueError(f"bio/activation length mismatch: {bio_arr.shape[0]} vs {act_arr.shape[0]}")
    y_bio = targets_for_theta(bio_arr)
    y_act = targets_for_theta(act_arr)
    return np.concatenate([y_bio, y_act], axis=-1).astype(np.float32)


def parse_target_weights(
    weights: Sequence[float] | None,
    *,
    n_targets: int = N_SINDY_TARGETS,
) -> np.ndarray:
    """Normalize optional per-target loss weights to shape ``[n_targets]``."""
    if weights is None:
        return np.ones((int(n_targets),), dtype=np.float32)
    arr = np.asarray(weights, dtype=np.float32).reshape(-1)
    if arr.shape[0] != int(n_targets):
        raise ValueError(f"Expected {n_targets} target weights, got {arr.shape[0]}")
    return arr
