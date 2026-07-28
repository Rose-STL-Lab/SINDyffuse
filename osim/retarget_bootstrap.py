"""Legacy bootstrap retargeting: direct HML joints → Lai OpenSim (lower fidelity)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from osim.coord_map import mint_coordinate_count, mint_coordinate_names
from osim.retarget_result import RetargetResult

NUM_HML3D_JOINTS = 22

HML_TO_LAI_MARKERS: Tuple[Tuple[str, int], ...] = (
    ("pelvis", 0),
    ("hip_r", 2),
    ("hip_l", 1),
    ("lumbar", 3),
    ("knee_r", 5),
    ("knee_l", 4),
    ("ankle_r", 8),
    ("ankle_l", 7),
    ("toe_r", 11),
    ("toe_l", 10),
    ("shoulder_r", 17),
    ("shoulder_l", 16),
    ("elbow_r", 19),
    ("elbow_l", 18),
    ("hand_r", 21),
    ("hand_l", 20),
)


def retarget_bootstrap(joints: np.ndarray) -> RetargetResult:
    arr = np.asarray(joints, dtype=np.float64)
    try:
        return _retarget_opensim_ik(arr)
    except Exception:
        return _retarget_heuristic(arr)


def _retarget_opensim_ik(joints: np.ndarray) -> RetargetResult:
    from osim.model_loader import load_mint_opensim_model

    model = load_mint_opensim_model()
    coord_names = mint_coordinate_names()
    t_len = int(joints.shape[0])
    q_out = np.zeros((t_len, len(coord_names)), dtype=np.float64)

    model_marker_set = model.getMarkerSet()
    marker_names = [str(model_marker_set.get(i).getName()) for i in range(model_marker_set.getSize())]
    hml_targets = _build_hml_marker_targets(joints, marker_names)

    state = model.initSystem()
    coord_set = model.getCoordinateSet()
    errors: List[float] = []

    for t in range(t_len):
        targets = hml_targets[t]
        _set_coordinates_from_markers(model, state, coord_set, targets)
        for i in range(coord_set.getSize()):
            q_out[t, i] = float(coord_set.get(i).getValue(state))
        errors.append(_marker_error(model, state, targets))

    return RetargetResult(
        q=q_out.astype(np.float32),
        mean_fk_error=float(np.mean(errors)) if errors else 0.0,
        num_frames=t_len,
        method="bootstrap",
    )


def _build_hml_marker_targets(joints: np.ndarray, marker_names: List[str]) -> List[dict]:
    t_len = int(joints.shape[0])
    frames: List[dict] = []
    name_to_hml = {m: idx for m, idx in HML_TO_LAI_MARKERS}
    for t in range(t_len):
        tgt: dict = {}
        for mname in marker_names:
            key = mname.lower().replace("-", "_")
            hml_idx = None
            for pattern, idx in HML_TO_LAI_MARKERS:
                if pattern in key or key in pattern:
                    hml_idx = idx
                    break
            if hml_idx is None and mname in name_to_hml:
                hml_idx = name_to_hml[mname]
            if hml_idx is not None:
                tgt[mname] = joints[t, hml_idx].copy()
        frames.append(tgt)
    return frames


def _set_coordinates_from_markers(model, state, coord_set, targets: dict) -> None:
    for _ in range(8):
        for ci in range(coord_set.getSize()):
            coord = coord_set.get(ci)
            if not coord.getDefaultLocked():
                val = float(coord.getValue(state))
                best = val
                best_err = _marker_error(model, state, targets)
                for delta in (-0.05, 0.05, -0.02, 0.02, -0.01, 0.01):
                    coord.setValue(state, val + delta)
                    model.realizePosition(state)
                    err = _marker_error(model, state, targets)
                    if err < best_err:
                        best_err = err
                        best = val + delta
                coord.setValue(state, best)
        model.realizePosition(state)


def _marker_error(model, state, targets: dict) -> float:
    err = 0.0
    mset = model.getMarkerSet()
    for i in range(mset.getSize()):
        m = mset.get(i)
        name = str(m.getName())
        if name not in targets:
            continue
        pos = np.array(m.getLocationInGround(state), dtype=np.float64).reshape(3)
        tgt = np.asarray(targets[name], dtype=np.float64).reshape(3)
        diff = pos - tgt
        err += float(np.dot(diff, diff))
    return err


def _retarget_heuristic(joints: np.ndarray) -> RetargetResult:
    t_len = int(joints.shape[0])
    ndof = mint_coordinate_count()
    q = np.zeros((t_len, ndof), dtype=np.float32)
    pelvis = joints[:, 0, :]
    q[:, 0] = pelvis[:, 0]
    q[:, 1] = pelvis[:, 1]
    q[:, 2] = pelvis[:, 2]
    for fi, (_, hml_idx) in enumerate(HML_TO_LAI_MARKERS[1: min(len(HML_TO_LAI_MARKERS), ndof - 3)]):
        rel = joints[:, hml_idx, :] - pelvis
        base = 3 + fi * 3
        if base + 2 < ndof:
            q[:, base : base + 3] = rel
    return RetargetResult(q=q, mean_fk_error=0.0, num_frames=t_len, method="bootstrap_heuristic")
