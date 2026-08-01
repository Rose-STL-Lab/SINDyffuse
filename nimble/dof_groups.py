from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple
LIMIT_GROUP_SPECS: Tuple[Tuple[str, slice], ...] = (('limit_pelvis_rot', slice(0, 3)), ('limit_pelvis_trans', slice(3, 6)), ('limit_hip_r', slice(6, 9)), ('limit_leg_r', slice(9, 13)), ('limit_hip_l', slice(13, 16)), ('limit_leg_l', slice(16, 20)), ('limit_lumbar', slice(20, 23)), ('limit_arm_r', slice(23, 30)), ('limit_arm_l', slice(30, 37)))
SYMMETRY_PAIRS: Tuple[Tuple[str, int, int], ...] = (('sym_hip_flex', 6, 13), ('sym_hip_add', 7, 14), ('sym_hip_rot', 8, 15), ('sym_knee', 9, 16), ('sym_ankle', 10, 17), ('sym_mtp', 12, 19))
ID_GROUP_SPECS: Tuple[Tuple[str, slice], ...] = (('id_pelvis', slice(0, 6)), ('id_hip_r', slice(6, 9)), ('id_leg_r', slice(9, 13)), ('id_hip_l', slice(13, 16)), ('id_leg_l', slice(16, 20)), ('id_lumbar', slice(20, 23)), ('id_arm_r', slice(23, 30)), ('id_arm_l', slice(30, 37)))
ID_SYMMETRY_SPECS: Tuple[Tuple[str, slice, slice], ...] = (('id_sym_hip', slice(6, 9), slice(13, 16)), ('id_sym_knee', slice(9, 10), slice(16, 17)), ('id_sym_ankle', slice(10, 11), slice(17, 18)))

@dataclass(frozen=True)
class DofGroups:
    limit_names: Tuple[str, ...]
    symmetry_names: Tuple[str, ...]
    id_names: Tuple[str, ...]
    id_symmetry_names: Tuple[str, ...]

def dof_groups() -> DofGroups:
    return DofGroups(limit_names=tuple((n for n, _ in LIMIT_GROUP_SPECS)), symmetry_names=tuple((n for n, _, _ in SYMMETRY_PAIRS)), id_names=tuple((n for n, _ in ID_GROUP_SPECS)), id_symmetry_names=tuple((n for n, _, _ in ID_SYMMETRY_SPECS)))

def limit_group_slices() -> Dict[str, slice]:
    return {name: sl for name, sl in LIMIT_GROUP_SPECS}

def id_group_slices() -> Dict[str, slice]:
    return {name: sl for name, sl in ID_GROUP_SPECS}