from __future__ import annotations
from functools import lru_cache
from typing import Sequence, Tuple
import numpy as np
from nimble.channels import BIOMECH_COMPONENT_KEYS
from nimble.guidance import NimbleGuidanceConfig
from nimble.opensim_log import opensim_quiet
from nimble.muscle_b3d import MUSCLE_ACTIVATION_ROWS

N_BIO_TARGETS = len(BIOMECH_COMPONENT_KEYS)
N_MUSCLE_TARGETS = int(MUSCLE_ACTIVATION_ROWS)
N_SINDY_TARGETS = N_BIO_TARGETS + N_MUSCLE_TARGETS


@lru_cache(maxsize=1)
def muscle_channel_names() -> Tuple[str, ...]:
    from nimble.muscle_activation import muscle_names
    with opensim_quiet('Off'):
        return tuple(muscle_names())


@lru_cache(maxsize=1)
def sindy_target_keys() -> Tuple[str, ...]:
    return tuple(BIOMECH_COMPONENT_KEYS) + muscle_channel_names()


def default_physics_cfg(*, fps: float = 20.0, max_frames: int | None = None) -> NimbleGuidanceConfig:
    del fps
    t_max = 64 if max_frames is None else int(max_frames)
    return NimbleGuidanceConfig(max_physics_frames=t_max, physics_on_cpu=True)


def bio_matrix(q: np.ndarray, *, fps: float, guidance_cfg: NimbleGuidanceConfig | None = None) -> np.ndarray:
    """Soft OpenSim-keypoint bio proxy (nimblephysics removed). Returns [T, N_BIO]."""
    del guidance_cfg, fps
    from nimble.lai_features import keypoints_from_lai_q
    if q.ndim != 2:
        raise ValueError(f'Expected q [T, ndof], got {q.shape}')
    t = int(q.shape[0])
    kp = keypoints_from_lai_q(np.asarray(q, dtype=np.float64))
    bio = np.zeros((t, N_BIO_TARGETS), dtype=np.float32)
    feet_y = kp[:, [4, 5], 1]
    pelvis_y = kp[:, 0, 1]
    bio[:, 0] = np.clip(-feet_y.min(axis=1), 0.0, None).astype(np.float32)
    bio[:, 1] = pelvis_y.astype(np.float32)
    if t > 1:
        bio[1:, 2] = np.linalg.norm(np.diff(kp[:, 0, :], axis=0), axis=1).astype(np.float32)
    return bio


def targets_for_theta(bio: np.ndarray) -> np.ndarray:
    if bio.ndim != 2:
        raise ValueError(f'Expected targets [T,C], got {bio.shape}')
    if bio.shape[0] < 2:
        return bio[:0]
    return bio[:-1, :].astype(np.float32)


def build_sindy_targets(bio: np.ndarray, activations: np.ndarray) -> np.ndarray:
    bio_arr = np.asarray(bio, dtype=np.float32)
    act_arr = np.asarray(activations, dtype=np.float32)
    if bio_arr.ndim != 2 or bio_arr.shape[1] != N_BIO_TARGETS:
        raise ValueError(f'Expected bio [T, {N_BIO_TARGETS}], got {bio_arr.shape}')
    if act_arr.ndim != 2 or act_arr.shape[1] != N_MUSCLE_TARGETS:
        raise ValueError(f'Expected activations [T, {N_MUSCLE_TARGETS}], got {act_arr.shape}')
    if bio_arr.shape[0] != act_arr.shape[0]:
        raise ValueError(f'bio/activation length mismatch: {bio_arr.shape[0]} vs {act_arr.shape[0]}')
    y_bio = targets_for_theta(bio_arr)
    y_act = targets_for_theta(act_arr)
    return np.concatenate([y_bio, y_act], axis=-1).astype(np.float32)


def parse_target_weights(weights: Sequence[float] | None, *, n_targets: int = N_SINDY_TARGETS) -> np.ndarray:
    if weights is None:
        return np.ones((int(n_targets),), dtype=np.float32)
    arr = np.asarray(weights, dtype=np.float32).reshape(-1)
    if arr.shape[0] != int(n_targets):
        raise ValueError(f'Expected {n_targets} target weights, got {arr.shape[0]}')
    return arr
