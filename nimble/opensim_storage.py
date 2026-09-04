"""OpenSim Storage helpers (no CasADi)."""
from __future__ import annotations
from typing import Any, List, Tuple
import numpy as np


def storage_to_array(storage: Any) -> Tuple[np.ndarray, List[str]]:
    labels: List[str] = []
    for i in range(storage.getColumnLabels().size()):
        labels.append(storage.getColumnLabels().get(i))
    rows: List[List[float]] = []
    times: List[float] = []
    for i in range(storage.getSize()):
        sv = storage.getStateVector(i)
        d = sv.getData()
        times.append(float(sv.getTime()))
        rows.append([float(d.get(j)) for j in range(d.size())])
    data = np.asarray(rows, dtype=np.float64)
    if not labels:
        return (data, labels)
    if str(labels[0]).strip().lower() == 'time' and data.ndim == 2:
        if data.shape[1] == len(labels) - 1:
            data = np.column_stack([np.asarray(times, dtype=np.float64), data])
        elif data.shape[1] == len(labels):
            data[:, 0] = np.asarray(times, dtype=np.float64)
    return (data, labels)


# Legacy name used by moco / coordinate_tracking callers
_storage_to_array = storage_to_array
