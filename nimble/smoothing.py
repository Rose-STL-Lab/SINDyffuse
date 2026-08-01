from __future__ import annotations
from typing import Any, Dict
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as sp_signal

def sample_rate_hz(fps: int, sampling_frequency: float | None) -> float:
    return float(sampling_frequency if sampling_frequency is not None else fps)
_PELVIS_ROT_DOF_SLICE = slice(0, 3)
_PELVIS_TRANS_DOF_SLICE = slice(3, 6)

def unwrap_pose_angles(poses: np.ndarray) -> np.ndarray:
    x = np.asarray(poses, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        return x
    out = x.copy()
    if out.shape[1] > _PELVIS_ROT_DOF_SLICE.stop:
        out[:, _PELVIS_ROT_DOF_SLICE] = np.unwrap(out[:, _PELVIS_ROT_DOF_SLICE], axis=0)
    if out.shape[1] > _PELVIS_TRANS_DOF_SLICE.stop:
        out[:, _PELVIS_TRANS_DOF_SLICE.stop:] = np.unwrap(out[:, _PELVIS_TRANS_DOF_SLICE.stop:], axis=0)
    return out

def effective_pose_smooth_cutoff_hz(cutoff_hz: float, sample_rate_hz: float) -> float:
    nyq = 0.5 * sample_rate_hz
    cap = nyq * 0.499
    return float(np.clip(cutoff_hz, 0.1, max(cap - 0.001, 0.25 * nyq)))

def _fir_num_taps(sample_rate_hz: float, cutoff_hz: float, butterworth_order: int) -> int:
    sr = float(sample_rate_hz)
    fc = float(max(cutoff_hz, 0.1))
    order = int(max(1, butterworth_order))
    base = 2 * int(sr / fc) + 1 + 2 * max(0, order - 2) * 5
    num_taps = min(401, max(5, base))
    if num_taps % 2 == 0:
        num_taps += 1
    return num_taps

def apply_pose_smoothing_torch(poses: torch.Tensor, *, fps: int, sampling_frequency: float | None, smooth_poses: bool, smooth_cutoff_hz: float, smooth_butterworth_order: int) -> tuple[torch.Tensor, Dict[str, Any]]:
    sr = sample_rate_hz(fps, sampling_frequency)
    meta: Dict[str, Any] = {'enabled': bool(smooth_poses), 'method': 'none', 'cutoff_hz_requested': float(smooth_cutoff_hz), 'cutoff_hz_effective': None, 'sample_rate_hz': float(sr), 'butterworth_order': int(smooth_butterworth_order)}
    if not smooth_poses or poses.shape[0] < 2:
        meta['enabled'] = bool(smooth_poses and poses.shape[0] >= 2)
        return (poses, meta)
    eff_fc = effective_pose_smooth_cutoff_hz(smooth_cutoff_hz, sr)
    meta['cutoff_hz_effective'] = float(eff_fc)
    meta['method'] = 'fir_firwin_conv1d'
    num_taps = _fir_num_taps(sr, eff_fc, smooth_butterworth_order)
    pad = (num_taps - 1) // 2
    t_frames = int(poses.shape[0])
    if t_frames <= 2 * pad:
        meta['enabled'] = False
        meta['method'] = 'skipped_short_clip'
        return (poses, meta)
    w_np = sp_signal.firwin(num_taps, eff_fc, fs=sr)
    w_t = torch.as_tensor(w_np, dtype=poses.dtype, device=poses.device).view(1, 1, -1)
    orig_shape = tuple(poses.shape)
    x_flat = poses.reshape(t_frames, -1)
    n_channels = int(x_flat.shape[1])
    x = x_flat.transpose(0, 1).contiguous().unsqueeze(0)
    weight = w_t.expand(n_channels, 1, num_taps)
    pad = (num_taps - 1) // 2
    xpad = F.pad(x, (pad, pad), mode='reflect')
    y = F.conv1d(xpad, weight, bias=None, stride=1, padding=0, groups=n_channels)
    y = y.squeeze(0).transpose(0, 1).contiguous()
    smoothed = y.reshape(orig_shape)
    return (smoothed, meta)