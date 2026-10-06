"""LaiUhlrich2022 per-motion NPZ cache (replaces Nimble B3D)."""
from __future__ import annotations
import os
import tempfile
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple
import numpy as np
from nimble.lai_coord_map import LAI_CACHE_DOF_NAMES, LAI_NUM_DOFS
from nimble.muscle_b3d import MUSCLE_ACTIVATION_ROWS

CACHE_SCHEMA_VERSION = 1
OPENSIM_MODEL_NAME = 'LaiUhlrich2022'
SINDY_U_COLS = 10
SINDY_C_COLS = 5
GUIDANCE_BIO_COLS = 40
SIM_GRF_COLS = 18  # keep in sync with nimble.moco_segment.SIM_GRF_COLS

__all__ = [
    'CACHE_SCHEMA_VERSION',
    'GUIDANCE_BIO_COLS',
    'OPENSIM_MODEL_NAME',
    'SINDY_C_COLS',
    'SINDY_U_COLS',
    'cache_meta_path',
    'has_muscle_activations',
    'lai_cache_dir',
    'list_motion_ids',
    'mean_path',
    'motion_npz_path',
    'read_motion_npz',
    'std_path',
    'write_cache_meta',
    'write_motion_npz',
]


def lai_cache_dir(data_root: str | Path, *, subdir: str | None = None) -> Path:
    from common.paths import LAI_CACHE_SUBDIR
    name = str(subdir).strip() if subdir else LAI_CACHE_SUBDIR
    return Path(data_root).expanduser().resolve() / name


def motion_npz_path(cache_dir: str | Path, motion_id: str) -> Path:
    return Path(cache_dir) / f'{motion_id}.npz'


def mean_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / 'Mean.npy'


def std_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / 'Std.npy'


def cache_meta_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / 'cache_meta.json'


def list_motion_ids(cache_dir: str | Path) -> list[str]:
    root = Path(cache_dir)
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob('*.npz'))


def _as_tq(arr: np.ndarray, *, name: str, cols: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f'{name}: expected 2D, got {a.shape}')
    if a.shape[1] == cols:
        return np.ascontiguousarray(a)
    if a.shape[0] == cols:
        return np.ascontiguousarray(a.T)
    raise ValueError(f'{name}: expected *x{cols} or {cols}x*, got {a.shape}')


def write_motion_npz(
    path: str | Path,
    *,
    motion_id: str,
    q: np.ndarray,
    fps: float = 20.0,
    mass_kg: float = 70.0,
    height_m: float = 1.75,
    muscle_activations: np.ndarray | None = None,
    muscle_activation_mask: np.ndarray | None = None,
    sim_grf: np.ndarray | None = None,
    sindy_u: np.ndarray | None = None,
    sindy_c: np.ndarray | None = None,
    guidance_bio: np.ndarray | None = None,
    opensim_model: str = OPENSIM_MODEL_NAME,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write/overwrite one motion NPZ. ``q`` is [T, 31] or [31, T]."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    q_arr = _as_tq(q, name='q', cols=LAI_NUM_DOFS)
    t_len = int(q_arr.shape[0])

    if muscle_activations is None:
        act = np.zeros((t_len, MUSCLE_ACTIVATION_ROWS), dtype=np.float32)
    else:
        act = _as_tq(muscle_activations, name='muscle_activations', cols=MUSCLE_ACTIVATION_ROWS)
        if act.shape[0] != t_len:
            raise ValueError(f'muscle_activations T mismatch: {act.shape[0]} vs {t_len}')

    if muscle_activation_mask is None:
        mask = np.zeros((t_len,), dtype=np.float32)
    else:
        mask = np.asarray(muscle_activation_mask, dtype=np.float32).reshape(-1)
        if mask.shape[0] != t_len:
            raise ValueError(f'mask T mismatch: {mask.shape[0]} vs {t_len}')

    if sim_grf is None:
        grf = np.full((t_len, SIM_GRF_COLS), np.nan, dtype=np.float32)
    else:
        grf = _as_tq(sim_grf, name='sim_grf', cols=SIM_GRF_COLS)
        if grf.shape[0] != t_len:
            raise ValueError(f'sim_grf T mismatch: {grf.shape[0]} vs {t_len}')

    if sindy_u is None:
        u = np.zeros((t_len, SINDY_U_COLS), dtype=np.float32)
    else:
        u = _as_tq(sindy_u, name='sindy_u', cols=SINDY_U_COLS)
        if u.shape[0] != t_len:
            raise ValueError(f'sindy_u T mismatch: {u.shape[0]} vs {t_len}')

    if sindy_c is None:
        c = np.zeros((t_len, SINDY_C_COLS), dtype=np.float32)
    else:
        c = _as_tq(sindy_c, name='sindy_c', cols=SINDY_C_COLS)
        if c.shape[0] != t_len:
            raise ValueError(f'sindy_c T mismatch: {c.shape[0]} vs {t_len}')

    if guidance_bio is None:
        bio = np.zeros((t_len, GUIDANCE_BIO_COLS), dtype=np.float32)
    else:
        bio = _as_tq(guidance_bio, name='guidance_bio', cols=GUIDANCE_BIO_COLS)
        if bio.shape[0] != t_len:
            raise ValueError(f'guidance_bio T mismatch: {bio.shape[0]} vs {t_len}')

    payload: Dict[str, Any] = {
        'q': q_arr,
        'muscle_activations': act,
        'muscle_activation_mask': mask,
        'sim_grf': grf,
        'sindy_u': u,
        'sindy_c': c,
        'guidance_bio': bio,
        'fps': np.float32(fps),
        'mass_kg': np.float32(mass_kg),
        'height_m': np.float32(height_m),
        'schema_version': np.int32(CACHE_SCHEMA_VERSION),
        'num_dofs': np.int32(LAI_NUM_DOFS),
        'motion_id': np.asarray(str(motion_id)),
        'opensim_model': np.asarray(str(opensim_model)),
    }
    if extra:
        for k, v in extra.items():
            if k in payload:
                continue
            payload[str(k)] = np.asarray(v)
    # Same-directory replace preserves the previous readable IK/activation file
    # if compression or the worker fails, and readers never see partial archives.
    fd, name = tempfile.mkstemp(prefix=f'.{out.name}.', dir=out.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as fp:
            np.savez_compressed(fp, **payload)
            fp.flush()
            os.fsync(fp.fileno())
        with np.load(temporary, allow_pickle=False) as archive:
            for key in archive.files:
                archive[key]  # Validate archive CRCs before publication.
        os.replace(temporary, out)
    finally:
        temporary.unlink(missing_ok=True)
    return out


def read_motion_npz(path: str | Path, *, mmap: bool = False) -> Dict[str, Any]:
    """Load one motion NPZ. Returns float32 arrays with q as [T, 31]."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'Missing Lai cache NPZ: {p}')
    mmap_mode = 'r' if mmap else None
    with np.load(p, allow_pickle=False, mmap_mode=mmap_mode) as z:
        q = _as_tq(np.asarray(z['q']), name='q', cols=LAI_NUM_DOFS)
        t_len = int(q.shape[0])
        out: Dict[str, Any] = {
            'q': np.asarray(q, dtype=np.float32),
            'muscle_activations': _as_tq(np.asarray(z['muscle_activations']), name='muscle_activations', cols=MUSCLE_ACTIVATION_ROWS).astype(np.float32),
            'muscle_activation_mask': np.asarray(z['muscle_activation_mask'], dtype=np.float32).reshape(-1),
            'sim_grf': _as_tq(np.asarray(z['sim_grf']), name='sim_grf', cols=SIM_GRF_COLS).astype(np.float32),
            'sindy_u': _as_tq(np.asarray(z['sindy_u']), name='sindy_u', cols=SINDY_U_COLS).astype(np.float32),
            'sindy_c': _as_tq(np.asarray(z['sindy_c']), name='sindy_c', cols=SINDY_C_COLS).astype(np.float32),
            'guidance_bio': _as_tq(np.asarray(z['guidance_bio']), name='guidance_bio', cols=GUIDANCE_BIO_COLS).astype(np.float32) if 'guidance_bio' in z.files else np.zeros((t_len, GUIDANCE_BIO_COLS), dtype=np.float32),
            'fps': float(np.asarray(z['fps']).reshape(())),
            'mass_kg': float(np.asarray(z['mass_kg']).reshape(())),
            'height_m': float(np.asarray(z['height_m']).reshape(())),
            'schema_version': int(np.asarray(z['schema_version']).reshape(())) if 'schema_version' in z.files else 0,
            'motion_id': str(np.asarray(z['motion_id']).reshape(())),
            'opensim_model': str(np.asarray(z['opensim_model']).reshape(())) if 'opensim_model' in z.files else OPENSIM_MODEL_NAME,
            'path': str(p),
            'num_frames': t_len,
            'activation_diagnostics': json.loads(str(np.asarray(z['activation_diagnostics_json']).reshape(()))) if 'activation_diagnostics_json' in z.files else {},
        }
    if out['muscle_activation_mask'].shape[0] != t_len:
        raise ValueError(f'mask length {out["muscle_activation_mask"].shape[0]} != T {t_len} in {p}')
    return out


def has_muscle_activations(path: str | Path, *, atol: float = 1e-8) -> bool:
    data = read_motion_npz(path, mmap=True)
    act = data['muscle_activations']
    mask = data['muscle_activation_mask']
    if mask is not None and np.asarray(mask).sum() > 0:
        return True
    finite = np.isfinite(act)
    if not finite.any():
        return False
    return not bool(np.allclose(np.nan_to_num(act, nan=0.0), 0.0, atol=float(atol)))


def write_cache_meta(cache_dir: str | Path, *, fps: float = 20.0, extra: Mapping[str, Any] | None = None) -> Path:
    from nimble.muscle_activation import muscle_names
    meta: Dict[str, Any] = {
        'schema_version': CACHE_SCHEMA_VERSION,
        'opensim_model': OPENSIM_MODEL_NAME,
        'num_dofs': LAI_NUM_DOFS,
        'dof_names': list(LAI_CACHE_DOF_NAMES),
        'num_muscles': MUSCLE_ACTIVATION_ROWS,
        'muscle_names': list(muscle_names()),
        'sim_grf_cols': SIM_GRF_COLS,
        'sindy_u_cols': SINDY_U_COLS,
        'sindy_c_cols': SINDY_C_COLS,
        'fps': float(fps),
    }
    if extra:
        meta.update(dict(extra))
    path = cache_meta_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    return path
