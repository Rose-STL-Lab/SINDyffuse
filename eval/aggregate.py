from __future__ import annotations
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
from eval.protocol import BOOTSTRAP_REPLICATES

def sem(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))

def bootstrap_ci95(values: Sequence[float], *, n_replicates: int=BOOTSTRAP_REPLICATES, seed: int=42) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (float('nan'), float('nan'))
    if arr.size == 1:
        v = float(arr[0])
        return (v, v)
    rng = np.random.default_rng(int(seed))
    n = int(arr.size)
    reps = int(n_replicates)
    samples = arr[rng.integers(0, n, size=(reps, n))].mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return (float(lo), float(hi))

def summarize_with_sem(metrics_by_caption: Dict[str, float]) -> Dict[str, float]:
    values = list(metrics_by_caption.values())
    mean = float(np.mean(values)) if values else float('nan')
    return {'mean': mean, 'sem': sem(values)}

def summarize_with_bootstrap(values: Sequence[float], *, n_replicates: int=BOOTSTRAP_REPLICATES, seed: int=42) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr)) if arr.size else float('nan')
    lo, hi = bootstrap_ci95(arr, n_replicates=n_replicates, seed=seed)
    return {'mean': mean, 'ci95_lo': lo, 'ci95_hi': hi}

def aggregate_caption_metrics(caption_values: Iterable[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys: List[str] = []
    for row in caption_values:
        for key in row:
            if key not in keys:
                keys.append(key)
    out: Dict[str, Dict[str, float]] = {}
    for key in keys:
        vals = [float(row[key]) for row in caption_values if key in row]
        out[key] = summarize_with_sem({str(i): v for i, v in enumerate(vals)})
    return out