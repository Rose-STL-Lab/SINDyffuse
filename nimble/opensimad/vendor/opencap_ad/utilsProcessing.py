"""Minimal OpenCap utilsProcessing stubs for SINDyffuse OpenSimAD."""
from __future__ import annotations
import numpy as np


def lowPassFilter(time, data, lowpass_cutoff_frequency, order=4):
    from scipy import signal
    if lowpass_cutoff_frequency is None or lowpass_cutoff_frequency <= 0:
        return data
    time = np.asarray(time, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    if time.size < 4:
        return data
    dt = float(np.median(np.diff(time)))
    fs = 1.0 / max(dt, 1e-8)
    nyq = 0.5 * fs
    wn = min(float(lowpass_cutoff_frequency) / nyq, 0.99)
    b, a = signal.butter(int(order), wn, btype='low')
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    out = np.zeros_like(data)
    for c in range(data.shape[1]):
        out[:, c] = signal.filtfilt(b, a, data[:, c])
    return out


def segment_squats(*args, **kwargs):
    raise RuntimeError('segment_squats not used by SINDyffuse OpenSimAD path')


def segment_STS(*args, **kwargs):
    raise RuntimeError('segment_STS not used by SINDyffuse OpenSimAD path')


def adjust_muscle_wrapping(*args, **kwargs):
    # No-op: Rajagopal AD model already prepared offline.
    return None


def generate_model_with_contacts(*args, **kwargs):
    # No-op: contacts model is prepared by nimble.opensimad.model_prep.
    return None
