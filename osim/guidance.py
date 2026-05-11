from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from diffusion.motion_process import recover_from_ric
from common.paths import DEFAULT_MODEL_PATH
from osim.features import compute_features_from_hml3d
from osim.moco_runtime import moco_marker_track_feature_summary


@dataclass
class OsimGuidanceWeights:
    lambda_vel: float = 0.1
    lambda_acc: float = 0.1
    lambda_torque: float = 0.1
    lambda_jerk: float = 0.1
    lambda_effort: float = 0.1
    lambda_contact: float = 0.1


@dataclass
class OsimOracleConfig:
    # Per-frame reduction across time for each component.
    # Supported: mean | lse | cvar
    time_reduce: str = "mean"
    # Optional robustification on each scalar component.
    # Supported: none | huber | log1p | charbonnier
    robust: str = "huber"
    huber_delta: float = 10.0
    charbonnier_eps: float = 1e-3
    # For CVaR reduction: keep top-k proportion of frames.
    cvar_alpha: float = 0.1
    # For lse reduction: temperature for smooth-max behavior.
    lse_temperature: float = 10.0
    # Optional timestep weight schedule applied in train loop.
    # Supported: none | linear
    t_weight_schedule: str = "none"
    # Use numpy/OpenSim-adjacent oracle values for monitoring and optional FD modes.
    use_oracle_numpy: bool = True
    # Smoothing options for oracle numpy pipeline.
    smooth_poses: bool = True
    smooth_cutoff_hz: float = 6.0
    smooth_butterworth_order: int = 2
    # Physics oracle: always MocoTrack (torque-driven marker tracking +
    # dynamics-derived generalized forces; requires OpenSim with Moco).
    opensim_model_path: str | None = None
    # Caps sequence length forwarded into Moco inside the oracle (per batch clip).
    opensim_max_frames: int = 64
    moco_weld_toes: bool = True
    moco_markers_global_weight: float = 10.0
    moco_control_effort_weight: float = 0.1
    moco_mesh_interval: float | None = None
    moco_markers_lowpass_hz: float = 0.0
    moco_max_solver_iterations: int = 200
    moco_convergence_tolerance: float = 5e-2
    moco_constraint_tolerance: float = 5e-2


def _robustify(x: torch.Tensor, robust: str, huber_delta: float, charbonnier_eps: float) -> torch.Tensor:
    mode = str(robust).strip().lower()
    if mode == "none":
        return x
    if mode == "huber":
        return huber_delta * (torch.sqrt(1.0 + (x / max(huber_delta, 1e-8)).pow(2)) - 1.0)
    if mode == "log1p":
        return torch.log1p(torch.clamp_min(x, 0.0))
    if mode == "charbonnier":
        return torch.sqrt(x.pow(2) + max(charbonnier_eps, 1e-12) ** 2)
    raise ValueError(f"Unsupported robust mode: {robust!r}")


def _reduce_time(x: torch.Tensor, mode: str, cvar_alpha: float, lse_temperature: float) -> torch.Tensor:
    if x.ndim == 1:
        x = x[:, None]
    m = str(mode).strip().lower()
    if m == "mean":
        return x.mean()
    if m == "lse":
        temp = max(float(lse_temperature), 1e-6)
        z = x * temp
        return (torch.logsumexp(z, dim=0) - torch.log(torch.tensor(float(x.shape[0]), device=x.device))) .mean() / temp
    if m == "cvar":
        alpha = min(max(float(cvar_alpha), 1e-4), 1.0)
        k = max(1, int(round(alpha * float(x.shape[0]))))
        topk, _ = torch.topk(x, k=k, dim=0, largest=True, sorted=False)
        return topk.mean()
    raise ValueError(f"Unsupported time_reduce mode: {mode!r}")


def _robustify_np(x: np.ndarray, robust: str, huber_delta: float, charbonnier_eps: float) -> np.ndarray:
    mode = str(robust).strip().lower()
    if mode == "none":
        return x
    if mode == "huber":
        d = max(float(huber_delta), 1e-8)
        return d * (np.sqrt(1.0 + (x / d) ** 2) - 1.0)
    if mode == "log1p":
        return np.log1p(np.clip(x, 0.0, None))
    if mode == "charbonnier":
        e = max(float(charbonnier_eps), 1e-12)
        return np.sqrt(x * x + e * e)
    raise ValueError(f"Unsupported robust mode: {robust!r}")


def _reduce_time_np(x: np.ndarray, mode: str, cvar_alpha: float, lse_temperature: float) -> float:
    if x.ndim == 1:
        x = x[:, None]
    m = str(mode).strip().lower()
    if m == "mean":
        return float(np.mean(x))
    if m == "lse":
        temp = max(float(lse_temperature), 1e-6)
        z = x * temp
        zmax = np.max(z, axis=0, keepdims=True)
        lse = np.log(np.mean(np.exp(z - zmax), axis=0)) + zmax.squeeze(0)
        return float(np.mean(lse) / temp)
    if m == "cvar":
        alpha = min(max(float(cvar_alpha), 1e-4), 1.0)
        k = max(1, int(round(alpha * float(x.shape[0]))))
        part = np.partition(x, kth=max(0, x.shape[0] - k), axis=0)
        top = part[-k:, :]
        return float(np.mean(top))
    raise ValueError(f"Unsupported time_reduce mode: {mode!r}")


class DeterministicOsimGuidance:
    def __init__(
        self,
        data_root: str,
        fps: float = 20.0,
        weights: OsimGuidanceWeights | None = None,
        oracle: OsimOracleConfig | None = None,
    ):
        root = Path(data_root).expanduser()
        mean_np = root / "Mean.npy"
        std_np = root / "Std.npy"
        if not mean_np.is_file():
            raise FileNotFoundError(
                f"OSIM guidance needs HumanML3D stats at {mean_np}; "
                f"fix data_root (got {root!s}) or export Mean.npy alongside new_joint_vecs."
            )
        if not std_np.is_file():
            raise FileNotFoundError(
                f"OSIM guidance needs HumanML3D stats at {std_np}; fix data_root (got {root!s})."
            )

        self.fps = float(fps)
        self.weights = weights or OsimGuidanceWeights()
        self.oracle = oracle or OsimOracleConfig()
        self.mean = torch.tensor(np.load(mean_np).astype(np.float32))
        self.std = torch.tensor(np.load(std_np).astype(np.float32))

    def _per_component(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "vel": feats["vel_l2"],
            "acc": feats["acc_l2"],
            "torque": feats["torque_l2"],
            "jerk": feats["jerk_l2"],
            "effort": feats["effort_proxy_l1"],
            "contact_gap": 1.0 - feats["inferred_contact_active_lr"][:, :2].mean(dim=1, keepdim=True),
        }

    def _aggregate(self, comp: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in comp.items():
            rv = _robustify(
                v,
                robust=self.oracle.robust,
                huber_delta=float(self.oracle.huber_delta),
                charbonnier_eps=float(self.oracle.charbonnier_eps),
            )
            out[k] = _reduce_time(
                rv,
                mode=self.oracle.time_reduce,
                cvar_alpha=float(self.oracle.cvar_alpha),
                lse_temperature=float(self.oracle.lse_temperature),
            )
        return out

    def _oracle_components_numpy(self, joints_t: torch.Tensor) -> Dict[str, float]:
        b = int(joints_t.shape[0])
        dt = 1.0 / max(self.fps, 1e-6)
        vals = {
            "vel": 0.0,
            "acc": 0.0,
            "torque": 0.0,
            "jerk": 0.0,
            "effort": 0.0,
            "contact_gap": 0.0,
        }
        for i in range(b):
            poses = joints_t[i].detach().cpu().numpy().astype(np.float32)
            feats, _ = compute_features_from_hml3d(
                poses=poses,
                dt=float(dt),
                include_poses=False,
                model_path=None,
                fps=int(self.fps),
                sampling_frequency=float(self.fps),
                smooth_poses=bool(self.oracle.smooth_poses),
                smooth_cutoff_hz=float(self.oracle.smooth_cutoff_hz),
                smooth_butterworth_order=int(self.oracle.smooth_butterworth_order),
            )
            comp_np = {
                "vel": feats["vel_l2"],
                "acc": feats["acc_l2"],
                "torque": feats["torque_l2"],
                "jerk": feats["jerk_l2"],
                "effort": feats["effort_proxy_l1"],
                "contact_gap": 1.0 - np.mean(feats["inferred_contact_active_lr"][:, :2], axis=1, keepdims=True),
            }
            sdk = moco_marker_track_feature_summary(
                poses,
                model_path=str(self.oracle.opensim_model_path or DEFAULT_MODEL_PATH),
                dt=float(dt),
                fps=float(self.fps),
                max_frames=int(self.oracle.opensim_max_frames),
                smooth_before_track=bool(self.oracle.smooth_poses),
                smooth_cutoff_hz=float(self.oracle.smooth_cutoff_hz),
                smooth_butterworth_order=int(self.oracle.smooth_butterworth_order),
                weld_toes=bool(self.oracle.moco_weld_toes),
                markers_global_tracking_weight=float(self.oracle.moco_markers_global_weight),
                control_effort_weight=float(self.oracle.moco_control_effort_weight),
                mesh_interval=self.oracle.moco_mesh_interval,
                markers_lowpass_hz=float(self.oracle.moco_markers_lowpass_hz),
                max_solver_iterations=int(self.oracle.moco_max_solver_iterations),
                convergence_tolerance=float(self.oracle.moco_convergence_tolerance),
                constraint_tolerance=float(self.oracle.moco_constraint_tolerance),
            )
            comp_np["torque"] = sdk["moco_torque_l2"].reshape(-1, 1)
            comp_np["vel"] = sdk["moco_vel_proxy_l2"].reshape(-1, 1)
            comp_np["acc"] = sdk["moco_acc_proxy_l2"].reshape(-1, 1)
            comp_np["jerk"] = sdk["moco_jerk_proxy_l2"].reshape(-1, 1)
            comp_np["effort"] = sdk["moco_effort_l1"].reshape(-1, 1)
            for k, v in comp_np.items():
                rv = _robustify_np(
                    v.astype(np.float64),
                    robust=self.oracle.robust,
                    huber_delta=float(self.oracle.huber_delta),
                    charbonnier_eps=float(self.oracle.charbonnier_eps),
                )
                vals[k] += _reduce_time_np(
                    rv,
                    mode=self.oracle.time_reduce,
                    cvar_alpha=float(self.oracle.cvar_alpha),
                    lse_temperature=float(self.oracle.lse_temperature),
                )
        inv_b = 1.0 / max(float(b), 1.0)
        for k in vals:
            vals[k] *= inv_b
        return vals

    def loss_and_stats(self, motion_norm: torch.Tensor) -> tuple[torch.Tensor, Dict[str, Any]]:
        b, _, d = motion_norm.shape
        mean = self.mean.to(motion_norm.device).view(1, 1, d)
        std = self.std.to(motion_norm.device).view(1, 1, d)
        motion_denorm = motion_norm * std + mean
        joints = recover_from_ric(motion_denorm, joints_num=22)
        dt = 1.0 / max(self.fps, 1e-6)
        losses = []
        stats_sum = {
            "vel": 0.0,
            "acc": 0.0,
            "torque": 0.0,
            "jerk": 0.0,
            "effort": 0.0,
            "contact_gap": 0.0,
        }
        for i in range(b):
            feats, _ = compute_features_from_hml3d(
                joints[i],
                dt=float(dt),
                include_poses=False,
                model_path=None,
                fps=int(self.fps),
                sampling_frequency=float(self.fps),
                smooth_poses=bool(self.oracle.smooth_poses),
                smooth_cutoff_hz=float(self.oracle.smooth_cutoff_hz),
                smooth_butterworth_order=int(self.oracle.smooth_butterworth_order),
            )
            comp = self._aggregate(self._per_component(feats))
            l = (
                self.weights.lambda_vel * comp["vel"]
                + self.weights.lambda_acc * comp["acc"]
                + self.weights.lambda_torque * comp["torque"]
                + self.weights.lambda_jerk * comp["jerk"]
                + self.weights.lambda_effort * comp["effort"]
                + self.weights.lambda_contact * comp["contact_gap"]
            )
            losses.append(l)
            for k in stats_sum:
                stats_sum[k] += float(comp[k].detach().cpu().item())
        loss = torch.stack(losses).mean()
        inv_b = 1.0 / max(float(b), 1.0)
        stats = {
            "osim_vel": stats_sum["vel"] * inv_b,
            "osim_acc": stats_sum["acc"] * inv_b,
            "osim_torque": stats_sum["torque"] * inv_b,
            "osim_jerk": stats_sum["jerk"] * inv_b,
            "osim_effort": stats_sum["effort"] * inv_b,
            "osim_contact_gap": stats_sum["contact_gap"] * inv_b,
        }
        if bool(self.oracle.use_oracle_numpy):
            oracle_np = self._oracle_components_numpy(joints)
            stats.update(
                {
                    "osim_oracle_vel": float(oracle_np["vel"]),
                    "osim_oracle_acc": float(oracle_np["acc"]),
                    "osim_oracle_torque": float(oracle_np["torque"]),
                    "osim_oracle_jerk": float(oracle_np["jerk"]),
                    "osim_oracle_effort": float(oracle_np["effort"]),
                    "osim_oracle_contact_gap": float(oracle_np["contact_gap"]),
                }
            )
            stats["osim_oracle_scalar"] = (
                float(self.weights.lambda_vel) * float(oracle_np["vel"])
                + float(self.weights.lambda_acc) * float(oracle_np["acc"])
                + float(self.weights.lambda_torque) * float(oracle_np["torque"])
                + float(self.weights.lambda_jerk) * float(oracle_np["jerk"])
                + float(self.weights.lambda_effort) * float(oracle_np["effort"])
                + float(self.weights.lambda_contact) * float(oracle_np["contact_gap"])
            )
        return loss, stats

    def guidance_weight(self, t: torch.Tensor, total_timesteps: int) -> torch.Tensor:
        mode = str(self.oracle.t_weight_schedule).strip().lower()
        if mode == "none":
            return torch.ones((), device=t.device, dtype=torch.float32)
        if mode == "linear":
            # Near t=0 (denoised), apply full guidance. Near large t, down-weight.
            denom = max(float(total_timesteps - 1), 1.0)
            w = 1.0 - (t.float() / denom)
            return torch.clamp(w.mean(), min=0.0, max=1.0)
        raise ValueError(f"Unsupported t_weight_schedule mode: {self.oracle.t_weight_schedule!r}")

    def loss(self, motion_norm: torch.Tensor) -> torch.Tensor:
        loss, _ = self.loss_and_stats(motion_norm)
        return loss

