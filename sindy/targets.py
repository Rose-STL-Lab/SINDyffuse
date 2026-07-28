"""L_bio + muscle activation targets from Nimble q / B3D for SINDy training."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence, Tuple

import numpy as np
import torch

from nimble.channels import BIOMECH_COMPONENT_KEYS
from nimble.guidance import NimbleGuidanceConfig
from nimble.muscle_activation import muscle_names, opensim_quiet
from nimble.muscle_b3d import MUSCLE_ACTIVATION_ROWS
from nimble.physics import physics_from_q

N_BIO_TARGETS = len(BIOMECH_COMPONENT_KEYS)
N_MUSCLE_TARGETS = int(MUSCLE_ACTIVATION_ROWS)
N_SINDY_TARGETS = N_BIO_TARGETS + N_MUSCLE_TARGETS


@lru_cache(maxsize=1)
def muscle_channel_names() -> Tuple[str, ...]:
    """Ordered Rajagopal muscle names (80)."""
    with opensim_quiet("Off"):
        return tuple(muscle_names())


@lru_cache(maxsize=1)
def sindy_target_keys() -> Tuple[str, ...]:
    """All SINDy target channel names: L_bio + 80 muscle."""
    return tuple(BIOMECH_COMPONENT_KEYS) + muscle_channel_names()


def default_physics_cfg(
    *,
    fps: float = 20.0,
    max_frames: int | None = None,
) -> NimbleGuidanceConfig:
    """Physics config for L_bio extraction."""
    t_max = 64 if max_frames is None else int(max_frames)
    return NimbleGuidanceConfig(
        max_physics_frames=t_max,
        physics_on_cpu=True,
        smooth_poses=True,
        smooth_cutoff_hz=6.0,
        mass_kg=70.0,
        g_mps2=9.81,
        contact_height_thresh_m=0.06,
        contact_speed_thresh_mps=1.2,
    )


def bio_matrix(
    q: np.ndarray,
    *,
    fps: float,
    guidance_cfg: NimbleGuidanceConfig | None = None,
) -> np.ndarray:
    """Nimble ``q`` ``[T, ndof]`` → L_bio channels ``[T, C]`` (no IK)."""
    if q.ndim != 2:
        raise ValueError(f"Expected q [T, ndof], got {q.shape}")
    t = int(q.shape[0])
    cfg = guidance_cfg or default_physics_cfg(fps=fps, max_frames=t)
    x = torch.from_numpy(q.astype(np.float32))
    with torch.no_grad():
        comp = physics_from_q(
            x, guidance_cfg=cfg, dt=1.0 / max(float(fps), 1e-8), fps=float(fps)
        )
    cols = [comp[k].reshape(-1).cpu().numpy() for k in BIOMECH_COMPONENT_KEYS]
    bio = np.stack(cols, axis=-1).astype(np.float32)
    if bio.shape[0] != t:
        bio = bio[:t]
    return bio


def targets_for_theta(bio: np.ndarray) -> np.ndarray:
    """Align per-frame rows with SINDy theta length ``T-1`` (drop last frame)."""
    if bio.ndim != 2:
        raise ValueError(f"Expected targets [T,C], got {bio.shape}")
    if bio.shape[0] < 2:
        return bio[:0]
    return bio[:-1, :].astype(np.float32)


def build_sindy_targets(bio: np.ndarray, activations: np.ndarray) -> np.ndarray:
    """Concat bio ``[T, C_bio]`` and muscle activations ``[T, 80]`` → ``[T-1, C_bio+80]``."""
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
