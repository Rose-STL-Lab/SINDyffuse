"""Map Nimble Rajagopal ``q`` to OpenSim coordinate motion (``.mot``) for MocoTrack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import opensim as osim

# Stable Nimble DOF order for the bundled Rajagopal skeleton (37 DOFs).
RAJAGOPAL_NIMBLE_DOF_NAMES: Tuple[str, ...] = (
    "ground_pelvis_0",
    "ground_pelvis_1",
    "ground_pelvis_2",
    "ground_pelvis_3",
    "ground_pelvis_4",
    "ground_pelvis_5",
    "hip_r_0",
    "hip_r_1",
    "hip_r_2",
    "walker_knee_r_0",
    "ankle_r_0",
    "subtalar_r_0",
    "mtp_r_0",
    "hip_l_0",
    "hip_l_1",
    "hip_l_2",
    "walker_knee_l_0",
    "ankle_l_0",
    "subtalar_l_0",
    "mtp_l_0",
    "back_0",
    "back_1",
    "back_2",
    "acromial_r_0",
    "acromial_r_1",
    "acromial_r_2",
    "elbow_r_0",
    "radioulnar_r_0",
    "radius_hand_r_0",
    "radius_hand_r_1",
    "acromial_l_0",
    "acromial_l_1",
    "acromial_l_2",
    "elbow_l_0",
    "radioulnar_l_0",
    "radius_hand_l_0",
    "radius_hand_l_1",
)

# Nimble joint-DOF name -> OpenSim Coordinate name (Rajagopal 2015).
NIMBLE_TO_OPENSIM_COORD: Dict[str, str] = {
    "ground_pelvis_0": "pelvis_tilt",
    "ground_pelvis_1": "pelvis_list",
    "ground_pelvis_2": "pelvis_rotation",
    "ground_pelvis_3": "pelvis_tx",
    "ground_pelvis_4": "pelvis_ty",
    "ground_pelvis_5": "pelvis_tz",
    "hip_r_0": "hip_flexion_r",
    "hip_r_1": "hip_adduction_r",
    "hip_r_2": "hip_rotation_r",
    "walker_knee_r_0": "knee_angle_r",
    "ankle_r_0": "ankle_angle_r",
    "subtalar_r_0": "subtalar_angle_r",
    "mtp_r_0": "mtp_angle_r",
    "hip_l_0": "hip_flexion_l",
    "hip_l_1": "hip_adduction_l",
    "hip_l_2": "hip_rotation_l",
    "walker_knee_l_0": "knee_angle_l",
    "ankle_l_0": "ankle_angle_l",
    "subtalar_l_0": "subtalar_angle_l",
    "mtp_l_0": "mtp_angle_l",
    "back_0": "lumbar_extension",
    "back_1": "lumbar_bending",
    "back_2": "lumbar_rotation",
    "acromial_r_0": "arm_flex_r",
    "acromial_r_1": "arm_add_r",
    "acromial_r_2": "arm_rot_r",
    "elbow_r_0": "elbow_flex_r",
    "radioulnar_r_0": "pro_sup_r",
    "radius_hand_r_0": "wrist_flex_r",
    "radius_hand_r_1": "wrist_dev_r",
    "acromial_l_0": "arm_flex_l",
    "acromial_l_1": "arm_add_l",
    "acromial_l_2": "arm_rot_l",
    "elbow_l_0": "elbow_flex_l",
    "radioulnar_l_0": "pro_sup_l",
    "radius_hand_l_0": "wrist_flex_l",
    "radius_hand_l_1": "wrist_dev_l",
}

# Rajagopal patellofemoral coupled coordinates (not in Nimble q); parent knee coords.
OPENSIM_COUPLED_BETA_FROM_KNEE: Dict[str, str] = {
    "knee_angle_r_beta": "knee_angle_r",
    "knee_angle_l_beta": "knee_angle_l",
}


@dataclass(frozen=True)
class RajagopalCoordMapping:
    """Index mapping between Nimble ``q`` and OpenSim coordinate order."""

    nimble_dof_names: Tuple[str, ...]
    opensim_coord_names: Tuple[str, ...]
    nimble_to_opensim_idx: Tuple[int, ...]
    rotational_coord_mask: Tuple[bool, ...]

    @property
    def num_nimble_dofs(self) -> int:
        return len(self.nimble_dof_names)

    @property
    def num_opensim_coords(self) -> int:
        return len(self.opensim_coord_names)


def build_rajagopal_coord_mapping(model_path: str | Path | None = None) -> RajagopalCoordMapping:
    """Build mapping using the bundled Rajagopal ``.osim`` coordinate set."""
    if model_path is None:
        import nimblephysics as nimble

        model_path = (
            Path(nimble.__file__).parent / "models" / "rajagopal_data" / "Rajagopal2015.osim"
        )
    model = osim.Model(str(model_path))
    model.initSystem()
    coord_set = model.getCoordinateSet()
    opensim_names: List[str] = [coord_set.get(i).getName() for i in range(coord_set.getSize())]
    coord_idx = {n: i for i, n in enumerate(opensim_names)}
    rotational = []
    for i in range(coord_set.getSize()):
        rotational.append(coord_set.get(i).getMotionType() == osim.Coordinate.Rotational)

    nimble_to_idx: List[int] = []
    for nn in RAJAGOPAL_NIMBLE_DOF_NAMES:
        on = NIMBLE_TO_OPENSIM_COORD[nn]
        if on not in coord_idx:
            raise KeyError(f"OpenSim coordinate {on!r} missing from model")
        nimble_to_idx.append(coord_idx[on])

    for beta_name in OPENSIM_COUPLED_BETA_FROM_KNEE:
        if beta_name not in coord_idx:
            raise KeyError(f"OpenSim beta coordinate {beta_name!r} missing from model")

    return RajagopalCoordMapping(
        nimble_dof_names=RAJAGOPAL_NIMBLE_DOF_NAMES,
        opensim_coord_names=tuple(opensim_names),
        nimble_to_opensim_idx=tuple(nimble_to_idx),
        rotational_coord_mask=tuple(rotational),
    )


def q_to_opensim_coordinates(
    q: np.ndarray,
    mapping: RajagopalCoordMapping | None = None,
) -> np.ndarray:
    """Convert Nimble ``q`` ``[T, 37]`` (radians) to OpenSim coords ``[T, 39]`` (deg for rotations)."""
    arr = np.asarray(q, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected q [T, ndof], got {arr.shape}")
    if arr.shape[1] != len(RAJAGOPAL_NIMBLE_DOF_NAMES):
        raise ValueError(
            f"Expected q [T, {len(RAJAGOPAL_NIMBLE_DOF_NAMES)}], got {arr.shape}"
        )
    m = mapping or build_rajagopal_coord_mapping()
    t_len = int(arr.shape[0])
    out = np.zeros((t_len, m.num_opensim_coords), dtype=np.float64)
    name_to_col = {n: i for i, n in enumerate(m.opensim_coord_names)}
    beta_from_knee = {
        name_to_col[beta]: name_to_col[parent]
        for beta, parent in OPENSIM_COUPLED_BETA_FROM_KNEE.items()
    }
    for t in range(t_len):
        for ni, oi in enumerate(m.nimble_to_opensim_idx):
            val = float(arr[t, ni])
            if m.rotational_coord_mask[oi]:
                val = float(np.rad2deg(val))
            out[t, oi] = val
        for beta_col, knee_col in beta_from_knee.items():
            out[t, beta_col] = float(out[t, knee_col])
    return out


def build_moco_states_table_processor(
    mot_path: str | Path,
    *,
    lowpass_hz: float = 6.0,
) -> osim.TableProcessor:
    """TableProcessor for MocoTrack: optional low-pass, absolute states, coupled betas."""
    tp = osim.TableProcessor(str(mot_path))
    if float(lowpass_hz) > 0.0:
        tp.append(osim.TabOpLowPassFilter(float(lowpass_hz)))
    tp.append(osim.TabOpUseAbsoluteStateNames())
    tp.append(osim.TabOpAppendCoupledCoordinateValues())
    return tp


def write_coordinates_mot(
    q: np.ndarray,
    mot_path: str | Path,
    *,
    fps: float,
    mapping: RajagopalCoordMapping | None = None,
) -> Path:
    """Write OpenSim coordinate ``.mot`` from Nimble ``q`` ``[T, ndof]``."""
    path = Path(mot_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = q_to_opensim_coordinates(q, mapping=mapping)
    m = mapping or build_rajagopal_coord_mapping()
    t_len = int(coords.shape[0])
    dt = 1.0 / max(float(fps), 1e-8)
    ncol = 1 + m.num_opensim_coords
    with path.open("w", encoding="utf-8") as f:
        f.write("Coordinates\n")
        f.write("version=1\n")
        f.write(f"nRows={t_len}\n")
        f.write(f"nColumns={ncol}\n")
        f.write("inDegrees=yes\n\n")
        f.write("endheader\n")
        f.write("time\t" + "\t".join(m.opensim_coord_names) + "\n")
        for t in range(t_len):
            row = [t * dt] + coords[t].tolist()
            f.write("\t".join(f"{x:.8f}" for x in row) + "\n")
    return path


def validate_coordinate_mapping(
    q_sample: np.ndarray,
    *,
    model_path: str | Path | None = None,
    atol: float = 1e-3,
) -> Dict[str, float]:
    """Sanity-check: round-trip Nimble positions through OpenSim state for one frame."""
    import nimblephysics as nimble
    from nimblephysics.models import rajagopal as raj

    q_row = np.asarray(q_sample, dtype=np.float64).reshape(-1)
    if q_row.shape[0] != len(RAJAGOPAL_NIMBLE_DOF_NAMES):
        raise ValueError(f"Expected q sample length {len(RAJAGOPAL_NIMBLE_DOF_NAMES)}")

    mapping = build_rajagopal_coord_mapping(model_path=model_path)
    osim_coords = q_to_opensim_coordinates(q_row[None, :], mapping=mapping)[0]

    if model_path is None:
        model_path = (
            Path(nimble.__file__).parent / "models" / "rajagopal_data" / "Rajagopal2015.osim"
        )
    model = osim.Model(str(model_path))
    model.initSystem()
    state = model.initSystem()
    coord_set = model.getCoordinateSet()
    for i in range(coord_set.getSize()):
        c = coord_set.get(i)
        c.setValue(state, float(osim_coords[i]))
    model.realizePosition(state)

    parsed = raj.RajagopalHumanBodyModel()
    sk = parsed.skeleton
    sk.setPositions(q_row)
    nimble_after = sk.getPositions().copy()

    sk.setPositions(np.zeros(sk.getNumDofs()))
    for ni, oi in enumerate(mapping.nimble_to_opensim_idx):
        val = float(osim_coords[oi])
        if mapping.rotational_coord_mask[oi]:
            val = float(np.deg2rad(val))
        sk.setPosition(ni, val)
    nimble_from_osim = sk.getPositions()

    err = float(np.max(np.abs(nimble_from_osim - q_row)))
    ok = err <= float(atol)
    return {
        "max_abs_error": err,
        "passed": float(ok),
        "num_opensim_coords": float(mapping.num_opensim_coords),
        "num_nimble_dofs": float(mapping.num_nimble_dofs),
    }


def nimble_dof_names() -> Tuple[str, ...]:
    """Ordered Nimble DOF names for Rajagopal (37)."""
    return RAJAGOPAL_NIMBLE_DOF_NAMES


def opensim_coordinate_names(model_path: str | Path | None = None) -> Tuple[str, ...]:
    """Ordered OpenSim coordinate names for Rajagopal (39)."""
    return build_rajagopal_coord_mapping(model_path=model_path).opensim_coord_names
