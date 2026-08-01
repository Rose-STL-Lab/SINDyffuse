from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from nimble.channels import build_channels
from nimble.ops import sanitize_joint_positions_torch
from nimble.ik import clear_body_ik_cache
from nimble.rajagopal_kin import clear_rajagopal_kin_cache, foot_body_indices, keypoints_torch
from nimble.smoothing import apply_pose_smoothing_torch
from nimble.skeleton_registry import load_skeleton
NIMBLE_AVAILABLE = True

def load_model(*, with_geometry: bool=False) -> Any:
    try:
        parsed, _spec = load_skeleton('rajagopal', with_geometry=with_geometry)
        skeleton = parsed.skeleton
        neutral = np.zeros(skeleton.getNumDofs(), dtype=np.float64)
        skeleton.setPositions(neutral)
        return parsed
    except Exception as exc:
        raise RuntimeError('Could not load Rajagopal skeleton via nimblephysics') from exc

@dataclass
class _GuidanceFkCache:
    sk: Any
    ndof: int
    q_lo: np.ndarray
    q_hi: np.ndarray
    foot_indices: Tuple[int, int]
_GUIDANCE_FK_CACHE_SLOT: Optional[_GuidanceFkCache] = None

def clear_cache() -> None:
    global _GUIDANCE_FK_CACHE_SLOT
    _GUIDANCE_FK_CACHE_SLOT = None
    clear_body_ik_cache()
    clear_rajagopal_kin_cache()

def _get_guidance_fk_cache() -> _GuidanceFkCache:
    global _GUIDANCE_FK_CACHE_SLOT
    if _GUIDANCE_FK_CACHE_SLOT is None:
        parsed = load_model()
        sk = parsed.skeleton
        q_lo = np.asarray(sk.getPositionLowerLimits(), dtype=np.float64).reshape(-1)
        q_hi = np.asarray(sk.getPositionUpperLimits(), dtype=np.float64).reshape(-1)
        _GUIDANCE_FK_CACHE_SLOT = _GuidanceFkCache(sk=sk, ndof=int(sk.getNumDofs()), q_lo=q_lo, q_hi=q_hi, foot_indices=foot_body_indices(sk))
    return _GUIDANCE_FK_CACHE_SLOT

def _use_torch_fk(guidance_cfg: Any, device: torch.device) -> bool:
    backend = str(getattr(guidance_cfg, 'fk_backend', 'torch')).strip().lower()
    if backend == 'nimble':
        return False
    if backend != 'torch':
        raise ValueError(f'Unsupported fk_backend: {backend!r}')
    if bool(getattr(guidance_cfg, 'physics_on_cpu', False)):
        return False
    return device.type != 'cpu'

def _keypoints_from_q(q: torch.Tensor, guidance_cfg: Any) -> torch.Tensor:
    if _use_torch_fk(guidance_cfg, q.device):
        return keypoints_torch(q)
    from nimble.rajagopal_kin import keypoints_numpy
    sk = _get_guidance_fk_cache().sk
    kp = keypoints_numpy(sk, q.detach().cpu().numpy())
    return torch.from_numpy(kp).to(device=q.device, dtype=q.dtype)

def physics_from_q(q: torch.Tensor, *, guidance_cfg: Any, dt: float, fps: float, cache: _GuidanceFkCache | None=None) -> Dict[str, torch.Tensor]:
    if q.ndim != 2:
        raise ValueError(f'Expected q [T, ndof], got {tuple(q.shape)}')
    orig_device = q.device
    use_torch = _use_torch_fk(guidance_cfg, orig_device)
    q_work = q.float() if use_torch else q.cpu().float()
    t_max = int(guidance_cfg.max_physics_frames)
    t_used = max(2, min(int(q_work.shape[0]), max(2, t_max)))
    fq = q_work[:t_used]
    fk_cache = cache or _get_guidance_fk_cache()
    keypoints = _keypoints_from_q(fq, guidance_cfg)
    keypoints = sanitize_joint_positions_torch(keypoints)
    keypoints_sm, _ = apply_pose_smoothing_torch(keypoints, fps=int(round(fps)), sampling_frequency=float(fps), smooth_poses=bool(guidance_cfg.smooth_poses), smooth_cutoff_hz=float(guidance_cfg.smooth_cutoff_hz), smooth_butterworth_order=int(guidance_cfg.smooth_butterworth_order))
    foot_indices = fk_cache.foot_indices
    if use_torch:
        foot_indices = foot_body_indices(fk_cache.sk)
    out = build_channels(keypoints_sm, fq, sk=fk_cache.sk, foot_indices=foot_indices, q_lo=fk_cache.q_lo, q_hi=fk_cache.q_hi, dt=float(dt), guidance_cfg=guidance_cfg, use_torch_fk=use_torch)
    keep_cpu = bool(getattr(guidance_cfg, 'physics_on_cpu', True)) and (not use_torch)
    if orig_device.type != 'cpu' and (not keep_cpu):
        out = {k: v.to(orig_device) for k, v in out.items()}
    return out

def physics_from_q_batch(q_batch: torch.Tensor, *, guidance_cfg: Any, dt: float, fps: float, cache: _GuidanceFkCache | None=None) -> list[Dict[str, torch.Tensor]]:
    if q_batch.ndim != 3:
        raise ValueError(f'Expected q_batch [B, T, ndof], got {tuple(q_batch.shape)}')
    return [physics_from_q(q_batch[i], guidance_cfg=guidance_cfg, dt=dt, fps=fps, cache=cache) for i in range(int(q_batch.shape[0]))]