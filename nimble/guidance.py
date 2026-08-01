from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
from nimble.channels import BIOMECH_COMPONENT_KEYS, parse_biomech_weights
from common.paths import nimble_b3d_dir
from nimble.physics import physics_from_q, physics_from_q_batch

@dataclass
class NimbleGuidanceWeights:
    components: Dict[str, float] = field(default_factory=lambda: parse_biomech_weights(None))

    def weight(self, key: str) -> float:
        return float(self.components.get(key, 0.0))

@dataclass
class NimbleGuidanceConfig:
    time_reduce: str = 'mean'
    robust: str = 'huber'
    huber_delta: float = 10.0
    charbonnier_eps: float = 0.001
    cvar_alpha: float = 0.1
    lse_temperature: float = 10.0
    t_weight_schedule: str = 'none'
    smooth_poses: bool = True
    smooth_cutoff_hz: float = 6.0
    smooth_butterworth_order: int = 2
    max_physics_frames: int = 64
    mass_kg: float = 70.0
    g_mps2: float = 9.81
    contact_height_thresh_m: float = 0.06
    contact_speed_thresh_mps: float = 1.2
    contact_gate_sharpness: float = 15.0
    physics_on_cpu: bool = True
    physics_batch_cap: int = 0
    fk_backend: str = 'torch'

def _robustify(x: torch.Tensor, robust: str, huber_delta: float, charbonnier_eps: float) -> torch.Tensor:
    mode = str(robust).strip().lower()
    if mode == 'none':
        return x
    if mode == 'huber':
        return huber_delta * (torch.sqrt(1.0 + (x / max(huber_delta, 1e-08)).pow(2)) - 1.0)
    if mode == 'log1p':
        return torch.log1p(torch.clamp_min(x, 0.0))
    if mode == 'charbonnier':
        return torch.sqrt(x.pow(2) + max(charbonnier_eps, 1e-12) ** 2)
    raise ValueError(f'Unsupported robust mode: {robust!r}')

def _reduce_time(x: torch.Tensor, mode: str, cvar_alpha: float, lse_temperature: float) -> torch.Tensor:
    if x.ndim == 1:
        x = x[:, None]
    m = str(mode).strip().lower()
    if m == 'mean':
        return x.mean()
    if m == 'lse':
        temp = max(float(lse_temperature), 1e-06)
        z = x * temp
        return (torch.logsumexp(z, dim=0) - torch.log(torch.tensor(float(x.shape[0]), device=x.device))).mean() / temp
    if m == 'cvar':
        alpha = min(max(float(cvar_alpha), 0.0001), 1.0)
        k = max(1, int(round(alpha * float(x.shape[0]))))
        topk, _ = torch.topk(x, k=k, dim=0, largest=True, sorted=False)
        return topk.mean()
    raise ValueError(f'Unsupported time_reduce mode: {mode!r}')

class DeterministicNimbleGuidance:

    def __init__(self, data_root: str, fps: float=20.0, weights: NimbleGuidanceWeights | None=None, nimble_settings: NimbleGuidanceConfig | None=None):
        root = Path(data_root).expanduser()
        cache = nimble_b3d_dir(root)
        if not cache.is_dir():
            raise FileNotFoundError(f'Nimble guidance requires Nimble B3D cache at {cache}. Run preprocess_nimble.py first.')
        mean_np = cache / 'Mean.npy'
        std_np = cache / 'Std.npy'
        if not mean_np.is_file() or not std_np.is_file():
            raise FileNotFoundError(f'Nimble guidance needs Mean.npy and Std.npy at {root}. Run preprocess_nimble.py first.')
        self.fps = float(fps)
        self.weights = weights or NimbleGuidanceWeights()
        self.nimble_settings = nimble_settings or NimbleGuidanceConfig()
        self.mean = torch.tensor(np.load(mean_np).astype(np.float32))
        self.std = torch.tensor(np.load(std_np).astype(np.float32))

    def _aggregate(self, comp: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in comp.items():
            rv = _robustify(v, robust=self.nimble_settings.robust, huber_delta=float(self.nimble_settings.huber_delta), charbonnier_eps=float(self.nimble_settings.charbonnier_eps))
            out[k] = _reduce_time(rv, mode=self.nimble_settings.time_reduce, cvar_alpha=float(self.nimble_settings.cvar_alpha), lse_temperature=float(self.nimble_settings.lse_temperature))
        return out

    def _weighted_scalar(self, comp: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = torch.zeros((), device=next(iter(comp.values())).device, dtype=next(iter(comp.values())).dtype)
        for key in BIOMECH_COMPONENT_KEYS:
            if key not in comp:
                continue
            w = self.weights.weight(key)
            if w != 0.0:
                total = total + w * comp[key]
        return total

    def loss_and_stats(self, motion_norm: torch.Tensor) -> tuple[torch.Tensor, Dict[str, Any]]:
        b, _, d = motion_norm.shape
        out_device = motion_norm.device
        use_cpu_physics = bool(getattr(self.nimble_settings, 'physics_on_cpu', False))
        motion_work = motion_norm.cpu() if use_cpu_physics else motion_norm
        mean = self.mean.to(motion_work.device).view(1, 1, d)
        std = self.std.to(motion_work.device).view(1, 1, d)
        motion_denorm = motion_work * std + mean
        dt = 1.0 / max(self.fps, 1e-06)
        losses = []
        stats_sum = {k: 0.0 for k in BIOMECH_COMPONENT_KEYS}
        cap = int(getattr(self.nimble_settings, 'physics_batch_cap', 0) or 0)
        b_phys = min(b, cap) if cap > 0 else b
        batch_denorm = motion_denorm[:b_phys].contiguous()
        comp_list = physics_from_q_batch(batch_denorm, guidance_cfg=self.nimble_settings, dt=float(dt), fps=float(self.fps))
        for comp in comp_list:
            comp = self._aggregate(comp)
            l = self._weighted_scalar(comp)
            losses.append(l)
            for k in BIOMECH_COMPONENT_KEYS:
                if k in comp:
                    stats_sum[k] += float(comp[k].detach().cpu().item())
        loss = torch.stack(losses).mean()
        if out_device.type != 'cpu':
            loss = loss.to(out_device)
        inv_b = 1.0 / max(float(b), 1.0)
        stats: Dict[str, Any] = {f'nimble_{k}': stats_sum[k] * inv_b for k in BIOMECH_COMPONENT_KEYS}
        stats['nimble_guidance_scalar'] = float(loss.detach().cpu().item())
        return (loss, stats)

    def guidance_weight(self, t: torch.Tensor, total_timesteps: int) -> torch.Tensor:
        mode = str(self.nimble_settings.t_weight_schedule).strip().lower()
        if mode == 'none':
            return torch.ones((), device=t.device, dtype=torch.float32)
        if mode == 'linear':
            denom = max(float(total_timesteps - 1), 1.0)
            w = 1.0 - t.float() / denom
            return torch.clamp(w.mean(), min=0.0, max=1.0)
        raise ValueError(f'Unsupported t_weight_schedule mode: {self.nimble_settings.t_weight_schedule!r}')

    def loss(self, motion_norm: torch.Tensor) -> torch.Tensor:
        loss, _ = self.loss_and_stats(motion_norm)
        return loss

def _split_nimble_guidance_cfg(nimble_cfg: Dict[str, Any] | None) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if not nimble_cfg:
        return ({}, {})
    wcfg = nimble_cfg.get('weights', {}) if isinstance(nimble_cfg.get('weights'), dict) else {}
    pcfg = dict(nimble_cfg['physics']) if isinstance(nimble_cfg.get('physics'), dict) else {}
    return (wcfg, pcfg)

def _physics_max_frames(pcfg: Dict[str, Any], window_frames: int | None) -> int:
    raw = int(pcfg.get('max_physics_frames', 64))
    max_frames = raw
    if window_frames is not None:
        max_frames = min(max_frames, max(2, int(window_frames)))
    return max_frames

def _cfg_float(pcfg: Dict[str, Any], key: str, default: float) -> float:
    return float(pcfg.get(key, default))

def _cfg_int(pcfg: Dict[str, Any], key: str, default: int) -> int:
    return int(pcfg.get(key, default))

def _cfg_bool(pcfg: Dict[str, Any], key: str, default: bool) -> bool:
    return bool(pcfg.get(key, default))

def build_nimble_guidance(data_root: str, fps: float, nimble_cfg: Dict[str, Any] | None=None, *, window_frames: int | None=None) -> DeterministicNimbleGuidance:
    wcfg, pcfg = _split_nimble_guidance_cfg(nimble_cfg)
    max_frames = _physics_max_frames(pcfg, window_frames)
    return DeterministicNimbleGuidance(data_root=data_root, fps=float(fps), weights=NimbleGuidanceWeights(components=parse_biomech_weights(wcfg)), nimble_settings=NimbleGuidanceConfig(time_reduce=str(pcfg.get('time_reduce', 'mean')), robust=str(pcfg.get('robust', 'huber')), huber_delta=_cfg_float(pcfg, 'huber_delta', 10.0), charbonnier_eps=_cfg_float(pcfg, 'charbonnier_eps', 0.001), cvar_alpha=_cfg_float(pcfg, 'cvar_alpha', 0.1), lse_temperature=_cfg_float(pcfg, 'lse_temperature', 10.0), t_weight_schedule=str(pcfg.get('t_weight_schedule', 'none')), smooth_poses=_cfg_bool(pcfg, 'smooth_poses', True), smooth_cutoff_hz=_cfg_float(pcfg, 'smooth_cutoff_hz', 6.0), smooth_butterworth_order=_cfg_int(pcfg, 'smooth_butterworth_order', 2), max_physics_frames=max_frames, mass_kg=_cfg_float(pcfg, 'mass_kg', 70.0), g_mps2=_cfg_float(pcfg, 'g_mps2', 9.81), contact_height_thresh_m=_cfg_float(pcfg, 'contact_height_thresh_m', 0.06), contact_speed_thresh_mps=_cfg_float(pcfg, 'contact_speed_thresh_mps', 1.2), contact_gate_sharpness=_cfg_float(pcfg, 'contact_gate_sharpness', 15.0), physics_on_cpu=_cfg_bool(pcfg, 'physics_on_cpu', False), physics_batch_cap=_cfg_int(pcfg, 'physics_batch_cap', 0), fk_backend=str(pcfg.get('fk_backend', 'torch')).strip().lower()))