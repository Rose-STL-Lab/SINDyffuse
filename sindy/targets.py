"""L_bio targets from Nimble q for SINDy training."""

from __future__ import annotations

import numpy as np
import torch

from nimble.channels import BIOMECH_COMPONENT_KEYS
from nimble.guidance import NimbleGuidanceConfig
from nimble.physics import physics_from_q


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
    """Align L_bio rows with SINDy theta length ``T-1`` (drop last frame)."""
    if bio.ndim != 2:
        raise ValueError(f"Expected bio [T,C], got {bio.shape}")
    if bio.shape[0] < 2:
        return bio[:0]
    return bio[:-1, :].astype(np.float32)
