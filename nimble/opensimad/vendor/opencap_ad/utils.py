"""Minimal OpenCap utils stubs for SINDyffuse OpenSimAD (no OpenCap cloud auth)."""
from __future__ import annotations
import numpy as np
import opensim


def storage_to_numpy(storage_file, excess_header_entries=0):
    table = opensim.TimeSeriesTable(str(storage_file))
    labels = list(table.getColumnLabels())
    times = np.array([float(t) for t in table.getIndependentColumn()], dtype=np.float64)
    data = table.getMatrix().to_numpy()
    out = {'time': times}
    for i, lab in enumerate(labels):
        out[str(lab)] = data[:, i]
    return out


def storage_to_dataframe(storage_file, headers):
    import pandas as pd
    raw = storage_to_numpy(storage_file)
    cols = {}
    for h in headers:
        if h in raw:
            cols[h] = raw[h]
        else:
            # try suffix match
            for k, v in raw.items():
                if k == h or k.endswith(h) or h in k:
                    cols[h] = v
                    break
    cols['time'] = raw['time']
    return pd.DataFrame(cols)


def numpy_to_storage(labels, data, storage_file, datatype=None):
    # Write a simple OpenSim .mot / storage file.
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    with open(storage_file, 'w', encoding='utf-8') as f:
        f.write('none\n')
        f.write('version=1\n')
        f.write(f'nRows={n}\n')
        f.write(f'nColumns={len(labels)}\n')
        f.write('inDegrees=no\n')
        f.write('endheader\n')
        f.write('\t'.join(labels) + '\n')
        for r in range(n):
            f.write('\t'.join(f'{float(data[r, c]):.8f}' for c in range(len(labels))) + '\n')


def download_kinematics(*args, **kwargs):
    raise RuntimeError('OpenCap download_kinematics is disabled in SINDyffuse; provide local IK .mot')


def import_metadata(filePath):
    import yaml
    with open(filePath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    # Normalize keys used by OpenCap OpenSimAD.
    if 'openSimModel' not in data and 'OpenSimModel' in data:
        data['openSimModel'] = data['OpenSimModel']
    data.setdefault('mass_kg', 70.0)
    data.setdefault('height_m', 1.75)
    return data
