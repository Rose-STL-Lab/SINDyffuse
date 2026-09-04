"""HumanML3D joint positions → LaiUhlrich2022 coordinates via OpenSim IK."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple
import numpy as np
import opensim as osim
from nimble.ik import fill_invalid_pose_frames, pose_is_invalid
from nimble.lai_coord_map import (
    LAI_NUM_DOFS,
    PELVIS_TY_COL,
    build_lai_coord_mapping,
    lai_model_path,
    opensim_coordinates_to_q,
    q_to_opensim_coordinates,
)
from nimble.opensim_log import opensim_quiet

# (marker_name, body_name, hml3d_joint_index)
HML3D_MARKER_MAP: Tuple[Tuple[str, str, int], ...] = (
    ('pelvis', 'pelvis', 0),
    ('hip_l', 'femur_l', 1),
    ('hip_r', 'femur_r', 2),
    ('torso', 'torso', 3),
    ('knee_l', 'tibia_l', 4),
    ('knee_r', 'tibia_r', 5),
    ('ankle_l', 'calcn_l', 7),
    ('ankle_r', 'calcn_r', 8),
    ('toe_l', 'toes_l', 10),
    ('toe_r', 'toes_r', 11),
    ('shoulder_l', 'humerus_l', 16),
    ('shoulder_r', 'humerus_r', 17),
    ('elbow_l', 'ulna_l', 18),
    ('elbow_r', 'ulna_r', 19),
    ('wrist_l', 'hand_l', 20),
    ('wrist_r', 'hand_r', 21),
)

_IK_MODEL_CACHE: Dict[str, osim.Model] = {}


def clear_opensim_ik_cache() -> None:
    _IK_MODEL_CACHE.clear()


def _ensure_geometry_search_path(model_path: Path) -> None:
    geom = model_path.parent / 'Geometry'
    if geom.is_dir():
        try:
            osim.ModelVisualizer.addDirToGeometrySearchPaths(str(geom))
        except Exception:
            pass


def _load_ik_model(model_path: Path) -> osim.Model:
    key = str(model_path.resolve())
    if key in _IK_MODEL_CACHE:
        return _IK_MODEL_CACHE[key]
    _ensure_geometry_search_path(model_path)
    model = osim.Model(str(model_path))
    for marker_name, body_name, _ in HML3D_MARKER_MAP:
        body = model.getBodySet().get(body_name)
        marker = osim.Marker()
        marker.setName(marker_name)
        marker.setParentFrame(body)
        marker.set_location(osim.Vec3(0.0, 0.0, 0.0))
        model.addMarker(marker)
    model.initSystem()
    cs = model.getCoordinateSet()
    for i in range(cs.getSize()):
        cs.get(i).set_locked(False)
    _IK_MODEL_CACHE[key] = model
    return model


def _apply_cache_q_to_state(model: osim.Model, state: Any, q_row: np.ndarray, mapping) -> None:
    coords_deg = q_to_opensim_coordinates(np.asarray(q_row, dtype=np.float64)[None, :], mapping=mapping)[0]
    cs = model.getCoordinateSet()
    name_to_i = {cs.get(i).getName(): i for i in range(cs.getSize())}
    for oi, name in enumerate(mapping.opensim_coord_names):
        if name not in name_to_i:
            continue
        val = float(coords_deg[oi])
        if mapping.rotational_coord_mask[oi]:
            val = float(np.deg2rad(val))
        cs.get(name_to_i[name]).setValue(state, val, False)


def _state_to_cache_q(model: osim.Model, state: Any, mapping) -> np.ndarray:
    cs = model.getCoordinateSet()
    name_to_i = {cs.get(i).getName(): i for i in range(cs.getSize())}
    coords_deg = np.zeros(mapping.num_opensim_coords, dtype=np.float64)
    for oi, name in enumerate(mapping.opensim_coord_names):
        if name not in name_to_i:
            continue
        val = float(cs.get(name_to_i[name]).getValue(state))
        if mapping.rotational_coord_mask[oi]:
            coords_deg[oi] = float(np.rad2deg(val))
        else:
            coords_deg[oi] = val
    return opensim_coordinates_to_q(coords_deg[None, :], mapping=mapping)[0]


def fit_q_lai(poses: np.ndarray, *, model_path: Path | None=None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fit HumanML3D joints [T,22,3] → cache q [ndof, T] (radians / meters)."""
    arr = np.asarray(poses, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (22, 3):
        raise ValueError(f'Expected poses [T,22,3], got {arr.shape}')
    t_frames = int(arr.shape[0])
    path = Path(model_path) if model_path is not None else lai_model_path()
    mapping = build_lai_coord_mapping(path)
    marker_names = tuple(m[0] for m in HML3D_MARKER_MAP)
    hml_indices = tuple(m[2] for m in HML3D_MARKER_MAP)
    n_m = len(marker_names)
    q_out = np.zeros((LAI_NUM_DOFS, t_frames), dtype=np.float64)
    fk_loss = np.full(t_frames, np.nan, dtype=np.float64)
    solver_err = np.full(t_frames, np.nan, dtype=np.float64)
    success = 0
    last_q: np.ndarray | None = None

    with opensim_quiet('Off'):
        model = _load_ik_model(path)
        for t in range(t_frames):
            targets = np.array([arr[t, int(j), :] for j in hml_indices], dtype=np.float64)
            table = osim.TimeSeriesTableVec3()
            labels = osim.StdVectorString()
            for name in marker_names:
                labels.append(name)
            table.setColumnLabels(labels)
            row = osim.RowVectorVec3(n_m)
            for i in range(n_m):
                row[i] = osim.Vec3(float(targets[i, 0]), float(targets[i, 1]), float(targets[i, 2]))
            table.appendRow(0.0, row)
            weights = osim.SetMarkerWeights()
            for name in marker_names:
                mw = osim.MarkerWeight(name, 1.0)
                weights.cloneAndAppend(mw)
            markers_ref = osim.MarkersReference(table, weights)
            coord_refs = osim.SimTKArrayCoordinateReference()
            ik = osim.InverseKinematicsSolver(model, markers_ref, coord_refs)
            state = model.initSystem()
            if last_q is not None:
                _apply_cache_q_to_state(model, state, last_q, mapping)
            try:
                ik.assemble(state)
                track_ret = ik.track(state)
                err = float(track_ret) if track_ret is not None else 0.0
                q_frame = _state_to_cache_q(model, state, mapping)
                if pose_is_invalid(q_frame) and last_q is not None:
                    q_frame = last_q.copy()
                else:
                    success += 1
                    last_q = q_frame.copy()
                q_out[:, t] = q_frame
                solver_err[t] = err
                model.realizePosition(state)
                sq = 0.0
                for i, name in enumerate(marker_names):
                    loc = model.getMarkerSet().get(name).getLocationInGround(state)
                    pred = np.array([loc.get(0), loc.get(1), loc.get(2)], dtype=np.float64)
                    d = pred - targets[i]
                    sq += float(np.dot(d, d))
                fk_loss[t] = sq / max(n_m, 1)
            except Exception as exc:
                if last_q is not None:
                    q_out[:, t] = last_q
                solver_err[t] = float('nan')
                fk_loss[t] = float('nan')
                if t == 0:
                    import sys
                    print(f'WARNING: OpenSim IK frame 0 failed: {exc}', file=sys.stderr, flush=True)

    q_out, filled = fill_invalid_pose_frames(q_out)
    stats: Dict[str, Any] = {
        'mean_fk_loss': float(np.nanmean(fk_loss)) if np.isfinite(fk_loss).any() else float('nan'),
        'mean_ik_error': float(np.nanmean(solver_err)) if np.isfinite(solver_err).any() else float('nan'),
        'success_ratio': float(success / max(t_frames, 1)),
        'total_frames': float(t_frames),
        'success_frames': float(success),
        'filled_invalid_frames': float(filled),
        'num_dofs': float(LAI_NUM_DOFS),
        'ik_backend': 'opensim_lai',
        'per_frame_fk_loss': fk_loss,
        'per_frame_loss': solver_err,
    }
    return (q_out, stats)


def body_world_positions(q_row: np.ndarray, body_names: Sequence[str], *, model_path: Path | None=None) -> np.ndarray:
    """Return [len(bodies), 3] world origins for one cache-q pose."""
    mapping = build_lai_coord_mapping(model_path)
    path = Path(model_path) if model_path is not None else lai_model_path()
    with opensim_quiet('Off'):
        model = osim.Model(str(path))
        state = model.initSystem()
        _apply_cache_q_to_state(model, state, np.asarray(q_row, dtype=np.float64), mapping)
        model.realizePosition(state)
        out = np.zeros((len(body_names), 3), dtype=np.float64)
        for bi, bname in enumerate(body_names):
            body = model.getBodySet().get(str(bname))
            p = body.getPositionInGround(state)
            out[bi] = (p.get(0), p.get(1), p.get(2))
    return out


def apply_ground_offset_lai_q(q: np.ndarray, *, sphere_offset_y_m: float=-0.02) -> Tuple[np.ndarray, float]:
    """Shift pelvis_ty so calcaneus contact spheres sit just above y=0."""
    q_arr = np.asarray(q, dtype=np.float64).copy()
    if q_arr.ndim != 2 or q_arr.shape[1] != LAI_NUM_DOFS:
        raise ValueError(f'Expected q [T, {LAI_NUM_DOFS}], got {q_arr.shape}')
    # Subsample for speed on long motions
    step = max(1, int(q_arr.shape[0] // 64))
    min_y = float('inf')
    for t in range(0, int(q_arr.shape[0]), step):
        feet = body_world_positions(q_arr[t], ('calcn_l', 'calcn_r'))
        min_y = min(min_y, float(feet[:, 1].min()) + float(sphere_offset_y_m))
    if not np.isfinite(min_y):
        return (q_arr, 0.0)
    shift = -float(min_y)
    q_arr[:, PELVIS_TY_COL] += shift
    return (q_arr, shift)
