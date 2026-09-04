"""LaiUhlrich2022 coordinate layout for cache + OpenSimAD .mot I/O."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import opensim as osim

# Independent DOFs stored in lai_cache NPZ / activation pipeline (MTP welded off; no wrists; no knee betas).
# Order matches MinT/OpenCap tracking coordinates in mint_settings.coordinates_toTrack.
LAI_CACHE_DOF_NAMES: Tuple[str, ...] = (
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l',
    'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
    'arm_flex_r', 'arm_add_r', 'arm_rot_r', 'elbow_flex_r', 'pro_sup_r',
    'arm_flex_l', 'arm_add_l', 'arm_rot_l', 'elbow_flex_l', 'pro_sup_l',
)
LAI_NUM_DOFS = len(LAI_CACHE_DOF_NAMES)
PELVIS_TY_COL = int(LAI_CACHE_DOF_NAMES.index('pelvis_ty'))
OPENSIM_COUPLED_BETA_FROM_KNEE: Dict[str, str] = {
    'knee_angle_r_beta': 'knee_angle_r',
    'knee_angle_l_beta': 'knee_angle_l',
}
_TRANSLATIONAL = frozenset({'pelvis_tx', 'pelvis_ty', 'pelvis_tz'})

@dataclass(frozen=True)
class LaiCoordMapping:
    cache_dof_names: Tuple[str, ...]
    opensim_coord_names: Tuple[str, ...]
    cache_to_opensim_idx: Tuple[int, ...]
    rotational_coord_mask: Tuple[bool, ...]

    @property
    def num_cache_dofs(self) -> int:
        return len(self.cache_dof_names)

    @property
    def num_opensim_coords(self) -> int:
        return len(self.opensim_coord_names)

def lai_model_path() -> Path:
    from nimble.opensimad.paths import lai_uhlrich_model_path
    return lai_uhlrich_model_path()

def build_lai_coord_mapping(model_path: str | Path | None=None) -> LaiCoordMapping:
    path = Path(model_path) if model_path is not None else lai_model_path()
    model = osim.Model(str(path))
    model.initSystem()
    coord_set = model.getCoordinateSet()
    opensim_names: List[str] = [coord_set.get(i).getName() for i in range(coord_set.getSize())]
    coord_idx = {n: i for i, n in enumerate(opensim_names)}
    rotational: List[bool] = []
    for i in range(coord_set.getSize()):
        rotational.append(coord_set.get(i).getMotionType() == osim.Coordinate.Rotational)
    cache_to_idx: List[int] = []
    for name in LAI_CACHE_DOF_NAMES:
        if name not in coord_idx:
            raise KeyError(f'OpenSim coordinate {name!r} missing from {path}')
        cache_to_idx.append(coord_idx[name])
    return LaiCoordMapping(
        cache_dof_names=LAI_CACHE_DOF_NAMES,
        opensim_coord_names=tuple(opensim_names),
        cache_to_opensim_idx=tuple(cache_to_idx),
        rotational_coord_mask=tuple(rotational),
    )

def q_to_opensim_coordinates(q: np.ndarray, mapping: LaiCoordMapping | None=None) -> np.ndarray:
    """Cache q [T, 31] in rad/m → full OpenSim coordinate table in degrees/m."""
    arr = np.asarray(q, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f'Expected q [T, ndof], got {arr.shape}')
    if arr.shape[1] != LAI_NUM_DOFS:
        raise ValueError(f'Expected q [T, {LAI_NUM_DOFS}], got {arr.shape}')
    m = mapping or build_lai_coord_mapping()
    t_len = int(arr.shape[0])
    out = np.zeros((t_len, m.num_opensim_coords), dtype=np.float64)
    name_to_col = {n: i for i, n in enumerate(m.opensim_coord_names)}
    beta_from_knee = {
        name_to_col[beta]: name_to_col[parent]
        for beta, parent in OPENSIM_COUPLED_BETA_FROM_KNEE.items()
        if beta in name_to_col and parent in name_to_col
    }
    for t in range(t_len):
        for cache_i, oi in enumerate(m.cache_to_opensim_idx):
            val = float(arr[t, cache_i])
            if m.rotational_coord_mask[oi]:
                val = float(np.rad2deg(val))
            out[t, oi] = val
        for beta_col, knee_col in beta_from_knee.items():
            out[t, beta_col] = float(out[t, knee_col])
        # MTP welded / unused: leave at 0 if present
        for mtp in ('mtp_angle_r', 'mtp_angle_l'):
            if mtp in name_to_col:
                out[t, name_to_col[mtp]] = 0.0
    return out

def opensim_coordinates_to_q(coords_deg: np.ndarray, mapping: LaiCoordMapping | None=None) -> np.ndarray:
    """Full OpenSim table (degrees) → cache q [T, 31] radians/m."""
    arr = np.asarray(coords_deg, dtype=np.float64)
    m = mapping or build_lai_coord_mapping()
    if arr.ndim != 2 or arr.shape[1] != m.num_opensim_coords:
        raise ValueError(f'Expected coords [T, {m.num_opensim_coords}], got {arr.shape}')
    out = np.zeros((arr.shape[0], LAI_NUM_DOFS), dtype=np.float64)
    for cache_i, oi in enumerate(m.cache_to_opensim_idx):
        val = arr[:, oi]
        if m.rotational_coord_mask[oi]:
            val = np.deg2rad(val)
        out[:, cache_i] = val
    return out

def write_coordinates_mot(q: np.ndarray, mot_path: str | Path, *, fps: float, mapping: LaiCoordMapping | None=None) -> Path:
    path = Path(mot_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = mapping or build_lai_coord_mapping()
    coords = q_to_opensim_coordinates(q, mapping=m)
    t_len = int(coords.shape[0])
    dt = 1.0 / max(float(fps), 1e-08)
    ncol = 1 + m.num_opensim_coords
    with path.open('w', encoding='utf-8') as f:
        f.write('Coordinates\n')
        f.write('version=1\n')
        f.write(f'nRows={t_len}\n')
        f.write(f'nColumns={ncol}\n')
        f.write('inDegrees=yes\n\n')
        f.write('endheader\n')
        f.write('time\t' + '\t'.join(m.opensim_coord_names) + '\n')
        for t in range(t_len):
            row = [t * dt] + coords[t].tolist()
            f.write('\t'.join((f'{x:.8f}' for x in row)) + '\n')
    return path

def unlock_lai_coordinates(model: osim.Model) -> None:
    cs = model.getCoordinateSet()
    for i in range(cs.getSize()):
        cs.get(i).set_locked(False)
