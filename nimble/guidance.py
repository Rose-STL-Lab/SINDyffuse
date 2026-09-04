"""Optional slow OpenSim CPU guidance for paper baselines (not the default train path)."""
from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Any, Dict
import numpy as np
import torch
from common.paths import lai_cache_dir
from nimble.lai_coord_map import LAI_NUM_DOFS
from nimble.lai_features import keypoints_from_lai_q


@dataclass
class OpenSimGuidanceConfig:
    time_reduce: str = 'mean'
    foot_height_weight: float = 1.0
    velocity_smooth_weight: float = 0.1
    max_physics_frames: int = 64
    # Accepted for API compatibility with old NimbleGuidanceConfig callers (ignored).
    physics_on_cpu: bool = True
    smooth_poses: bool = True
    smooth_cutoff_hz: float = 6.0
    mass_kg: float = 70.0
    g_mps2: float = 9.81
    contact_height_thresh_m: float = 0.06
    contact_speed_thresh_mps: float = 1.2
    fk_backend: str = 'opensim'
    robust: bool = False
    huber_delta: float = 1.0
    charbonnier_eps: float = 1e-3
    cvar_alpha: float = 0.1
    lse_temperature: float = 1.0
    t_weight_schedule: str = 'none'
    physics_batch_cap: int = 0

    @staticmethod
    def from_mapping(data: dict | None) -> 'OpenSimGuidanceConfig':
        if not data:
            return OpenSimGuidanceConfig()
        allowed = {f.name for f in fields(OpenSimGuidanceConfig)}
        return OpenSimGuidanceConfig(**{k: v for k, v in data.items() if k in allowed})


class OpenSimGuidance:
    """Kinematic soft constraints via OpenSim FK keypoints (CPU, intentionally slow)."""

    def __init__(
        self,
        data_root: str,
        fps: float = 20.0,
        opensim_cfg: OpenSimGuidanceConfig | None = None,
        window_frames: int = 64,
    ):
        self.fps = float(fps)
        self.cfg = opensim_cfg or OpenSimGuidanceConfig()
        self.nimble_settings = self.cfg  # alias for old log code
        self.window_frames = int(window_frames)
        cache = lai_cache_dir(data_root)
        mp, sp = cache / 'Mean.npy', cache / 'Std.npy'
        if not mp.is_file() or not sp.is_file():
            raise FileNotFoundError(f'Missing Mean/Std under {cache}')
        self.mean = torch.from_numpy(np.load(mp).astype(np.float32).reshape(-1))
        self.std = torch.from_numpy(np.maximum(np.load(sp).astype(np.float32).reshape(-1), 1e-8))
        if int(self.mean.numel()) != LAI_NUM_DOFS:
            raise ValueError(f'Expected Mean dim {LAI_NUM_DOFS}, got {self.mean.numel()}')

    def _denorm(self, motion_norm: torch.Tensor) -> np.ndarray:
        mean = self.mean.to(device=motion_norm.device, dtype=motion_norm.dtype)
        std = self.std.to(device=motion_norm.device, dtype=motion_norm.dtype)
        q = motion_norm * std + mean
        return q.detach().cpu().numpy().astype(np.float64)

    def loss_and_stats(self, motion_norm: torch.Tensor) -> tuple[torch.Tensor, Dict[str, Any]]:
        if motion_norm.ndim != 3:
            raise ValueError(f'Expected [B,T,D], got {tuple(motion_norm.shape)}')
        q_batch = self._denorm(motion_norm)
        losses = []
        foot_vals = []
        vel_vals = []
        for b in range(int(q_batch.shape[0])):
            q = q_batch[b]
            t_cap = min(int(q.shape[0]), int(self.cfg.max_physics_frames))
            q = q[:t_cap]
            kp = keypoints_from_lai_q(q)
            feet_y = kp[:, [4, 5], 1]
            foot_pen = float(np.mean(np.clip(-feet_y, 0.0, None) ** 2))
            if q.shape[0] > 2:
                vel = np.diff(q, axis=0)
                acc = np.diff(vel, axis=0)
                vel_pen = float(np.mean(acc ** 2))
            else:
                vel_pen = 0.0
            total = float(self.cfg.foot_height_weight) * foot_pen + float(self.cfg.velocity_smooth_weight) * vel_pen
            losses.append(total)
            foot_vals.append(foot_pen)
            vel_vals.append(vel_pen)
        loss = torch.tensor(float(np.mean(losses)), device=motion_norm.device, dtype=motion_norm.dtype)
        stats = {
            'opensim_foot_penalty': float(np.mean(foot_vals)),
            'opensim_vel_penalty': float(np.mean(vel_vals)),
            'opensim_guidance_scalar': float(loss.detach().cpu().item()),
        }
        return (loss, stats)

    def loss(self, motion_norm: torch.Tensor) -> torch.Tensor:
        return self.loss_and_stats(motion_norm)[0]


def build_opensim_guidance(
    *,
    data_root: str,
    fps: float = 20.0,
    opensim_cfg: dict | OpenSimGuidanceConfig | None = None,
    window_frames: int = 64,
) -> OpenSimGuidance:
    if isinstance(opensim_cfg, OpenSimGuidanceConfig):
        cfg = opensim_cfg
    else:
        cfg = OpenSimGuidanceConfig.from_mapping(opensim_cfg if isinstance(opensim_cfg, dict) else None)
    return OpenSimGuidance(data_root=data_root, fps=fps, opensim_cfg=cfg, window_frames=window_frames)


NimbleGuidanceConfig = OpenSimGuidanceConfig
NimbleGuidanceWeights = OpenSimGuidanceConfig
DeterministicNimbleGuidance = OpenSimGuidance


def build_nimble_guidance(*, data_root: str, fps: float = 20.0, nimble_cfg: dict | None = None, window_frames: int = 64) -> OpenSimGuidance:
    return build_opensim_guidance(data_root=data_root, fps=fps, opensim_cfg=nimble_cfg, window_frames=window_frames)
