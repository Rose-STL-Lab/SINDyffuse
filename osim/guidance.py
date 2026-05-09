from __future__ import annotations

from dataclasses import dataclass

import torch

from diffusion.motion_process import recover_from_ric
from osim.features_torch import compute_features_from_hml3d_torch


@dataclass
class OsimGuidanceWeights:
    lambda_vel: float = 0.1
    lambda_acc: float = 0.1
    lambda_torque: float = 0.1
    lambda_jerk: float = 0.1
    lambda_effort: float = 0.1
    lambda_contact: float = 0.1


class DeterministicOsimGuidance:
    def __init__(self, data_root: str, fps: float = 20.0, weights: OsimGuidanceWeights | None = None):
        import numpy as np

        self.fps = float(fps)
        self.weights = weights or OsimGuidanceWeights()
        self.mean = torch.tensor(np.load(f"{data_root}/Mean.npy").astype(np.float32))
        self.std = torch.tensor(np.load(f"{data_root}/Std.npy").astype(np.float32))

    def loss(self, motion_norm: torch.Tensor) -> torch.Tensor:
        b, _, d = motion_norm.shape
        mean = self.mean.to(motion_norm.device).view(1, 1, d)
        std = self.std.to(motion_norm.device).view(1, 1, d)
        motion_denorm = motion_norm * std + mean
        joints = recover_from_ric(motion_denorm, joints_num=22)
        dt = 1.0 / max(self.fps, 1e-6)
        losses = []
        for i in range(b):
            feats, _ = compute_features_from_hml3d_torch(joints[i], dt=dt, include_poses=False, fps=int(self.fps))
            l = (
                self.weights.lambda_vel * feats["vel_l2"].mean()
                + self.weights.lambda_acc * feats["acc_l2"].mean()
                + self.weights.lambda_torque * feats["torque_l2"].mean()
                + self.weights.lambda_jerk * feats["jerk_l2"].mean()
                + self.weights.lambda_effort * feats["effort_proxy_l1"].mean()
                + self.weights.lambda_contact * (1.0 - feats["inferred_contact_active_lr"][:, :2].mean())
            )
            losses.append(l)
        return torch.stack(losses).mean()

