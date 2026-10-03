"""Select a diagnostic solve that is inside the existing polynomial domain."""
from __future__ import annotations
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from nimble.opensimad.mint_settings import MINT_POLYNOMIAL_BOUNDS

def filtered_bounds_violations(q: np.ndarray, coordinate_names: tuple[str, ...], *, fps: float,
                               mesh_interval: float=0.02) -> dict:
    """Mirror OpenCap getIK → filterDataFrame → interpolateDataFrame → ROM check.

    q is radians/metres. No clipping or modification is performed. Rounded motion
    times match write_coordinates_mot and the vendored interpolation routine.
    """
    values = np.asarray(q, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(coordinate_names) or len(values) < 8:
        raise ValueError('Canary requires at least eight frames with matching coordinates')
    if not np.isfinite(values).all() or not np.isfinite(fps) or fps <= 0 or not np.isfinite(mesh_interval) or mesh_interval <= 0:
        raise ValueError('Canary coordinates, fps and mesh interval must be finite and valid')
    times = np.round(np.arange(len(values), dtype=np.float64) / fps, 6)
    fs = np.round(1 / np.mean(np.diff(times)), 6)
    normalized_cutoff = min(6 / (fs / 2), 0.999)
    b, a = signal.butter(2, normalized_cutoff, 'low')
    filtered = signal.filtfilt(b, a, values, axis=0, padtype='odd', padlen=3 * (max(len(a), len(b)) - 1))
    duration = (len(values) - 1) / fps
    mesh_density = int(round(1 / mesh_interval))
    count = int(round(duration * mesh_density, 2))
    if count < 2:
        raise ValueError('Canary window has fewer than two mesh points')
    mesh_times = np.linspace(0, np.round(duration, 6), count)
    tracked = interp1d(times, filtered, axis=0)(mesh_times)
    violations = {}
    for coordinate, bounds in MINT_POLYNOMIAL_BOUNDS.items():
        degrees = np.rad2deg(tracked[:, coordinate_names.index(coordinate)])
        low, high = float(np.min(degrees)), float(np.max(degrees))
        if low < bounds['min'] or high > bounds['max']:
            violations[coordinate] = {'min_degrees': low, 'max_degrees': high,
                                     'allowed_min_degrees': bounds['min'], 'allowed_max_degrees': bounds['max']}
    return violations