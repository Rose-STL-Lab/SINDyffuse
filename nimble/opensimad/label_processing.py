"""Gap-preserving seam smoothing and MinT-style tracking analysis (not gates)."""
from __future__ import annotations
import numpy as np

LABEL_PROCESSING_VERSION = 'overlap_crossfade_tracking_grf_v1'

def stitch_solve_windows(t_len, segments, values, successful, *, blend_frames):
    """Crossfade successful neighbors using their buffered solve predictions.

    A failed core stays NaN, even when a neighboring solve has buffer predictions
    for it. No extrapolation and no blending across a failed segment. MinT states
    that seams are smoothed; this linear overlap crossfade is our documented
    implementation, not a claim about their unpublished generation algorithm.
    """
    if not segments or len(segments) != len(values) or len(values) != len(successful):
        raise ValueError('Require matching nonempty segments, predictions and outcomes')
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    cols = arrays[0].shape[1]
    out = np.full((t_len, cols), np.nan, dtype=np.float64)
    for spec, arr, ok in zip(segments, arrays, successful):
        if arr.shape != (spec.solve_len, cols):
            raise ValueError(f'Segment {spec.index} solve-window shape mismatch')
        if ok:
            start = spec.core_start - spec.solve_start
            out[spec.core_start:spec.core_end] = arr[start:start + spec.core_len]
    blend = max(0, int(blend_frames))
    for i in range(len(segments) - 1):
        left, right = segments[i:i+2]
        if not blend or not successful[i] or not successful[i+1] or left.core_end != right.core_start:
            continue
        boundary = left.core_end
        # Bound by both neighboring cores so adjacent blends cannot overlap.
        lo = max(left.solve_start, right.solve_start, boundary - blend,
                 left.core_start + left.core_len // 2)
        hi = min(left.solve_end, right.solve_end, boundary + blend,
                 right.core_start + (right.core_len + 1) // 2)
        if hi <= lo:
            continue
        a = arrays[i][lo-left.solve_start:hi-left.solve_start]
        b = arrays[i+1][lo-right.solve_start:hi-right.solve_start]
        alpha = (np.arange(hi-lo, dtype=float) + 1) / (hi-lo + 1)
        finite = np.isfinite(a) & np.isfinite(b)
        combined = a * (1-alpha[:, None]) + b * alpha[:, None]
        out[lo:hi] = np.where(finite, combined, out[lo:hi])
    return out.astype(np.float32)

def tracking_metrics(simulated, reference, coordinates):
    """Coordinates are radians/metres; report rotations in degrees, translation m."""
    sim, ref = np.asarray(simulated, dtype=float), np.asarray(reference, dtype=float)
    if sim.shape != ref.shape or sim.ndim != 2 or sim.shape[1] != len(coordinates):
        raise ValueError('Tracking arrays must be matching [frames, coordinates]')
    rows = []
    for i, name in enumerate(coordinates):
        trans = name in ('pelvis_tx', 'pelvis_ty', 'pelvis_tz')
        error = sim[:, i] - ref[:, i]
        error = error[np.isfinite(error)]
        if not trans:
            error = np.rad2deg(error)
        rows.append({'coordinate': name, 'is_translational': trans, 'unit': 'm' if trans else 'deg',
                     'sample_count': int(len(error)), 'rmse': float(np.sqrt(np.mean(error**2))) if len(error) else None,
                     'max_abs': float(np.max(np.abs(error))) if len(error) else None})
    return summarize_tracking(rows)

def summarize_tracking(rows):
    trans = [r for r in rows if r['is_translational'] and r['rmse'] is not None]
    rot = [r for r in rows if not r['is_translational'] and r['rmse'] is not None]
    complete = bool(rows) and all(r['rmse'] is not None for r in rows)
    worst_trans = max(trans, key=lambda r:r['rmse']) if trans else None
    worst_rot = max(rot, key=lambda r:r['rmse']) if rot else None
    return {'per_coordinate': rows,
            'max_translational_rmse_m': worst_trans['rmse'] if worst_trans else None,
            'max_rotational_rmse_deg': worst_rot['rmse'] if worst_rot else None,
            'worst_translational_coordinate': worst_trans['coordinate'] if worst_trans else None,
            'worst_rotational_coordinate': worst_rot['coordinate'] if worst_rot else None,
            'mint_analysis_flags': {'rotational_rmse_lt_5deg': complete and bool(rot) and all(r['rmse'] < 5 for r in rot),
                                    'translational_rmse_lt_0_02m': complete and bool(trans) and all(r['rmse'] < .02 for r in trans)},
            'policy': 'Diagnostic flags from MinT analysis; not acceptance gates'}

def pool_tracking(metrics):
    accum = {}
    for item in metrics:
        for row in item.get('per_coordinate', []):
            if row['rmse'] is None or not row['sample_count']:
                continue
            entry = accum.setdefault(row['coordinate'], {**row, 'sum_sq': 0., 'sample_count': 0, 'max_abs': 0.})
            entry['sum_sq'] += row['rmse']**2 * row['sample_count']
            entry['sample_count'] += row['sample_count']
            entry['max_abs'] = max(entry['max_abs'], row['max_abs'])
    rows = []
    for entry in accum.values():
        entry['rmse'] = float(np.sqrt(entry.pop('sum_sq') / entry['sample_count']))
        rows.append(entry)
    return summarize_tracking(rows)

def set_grf_validity(grf):
    arr = np.asarray(grf, dtype=np.float32).copy()
    arr[:, 12] = arr[:, 1] + arr[:, 7]
    for target, start in ((13,0),(14,6),(15,3),(16,9)):
        arr[:, target] = np.linalg.norm(arr[:, start:start+3], axis=1)
    arr[:, 17] = np.isfinite(arr[:, :12]).all(axis=1).astype(np.float32)
    return arr