from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import opensim as osim
from nimble.muscle_activation import _storage_to_array
from nimble.rajagopal_coord_map import RajagopalCoordMapping, build_moco_states_table_processor, write_coordinates_mot

# Default per-coordinate RMSE thresholds for post-Moco quality gates.
DEFAULT_MAX_TRANSLATIONAL_COORD_RMSE_M = 0.02
DEFAULT_MAX_ROTATIONAL_COORD_RMSE_DEG = 5.0


def is_translational_coordinate(name: str) -> bool:
    text = str(name)
    short = text.rsplit('/', 1)[-1]
    if short == 'value' and text.count('/') >= 2:
        short = text.rsplit('/', 2)[-2]
    return any((short.endswith(suffix) for suffix in ('_tx', '_ty', '_tz')))


def _column_index_for_coordinate(labels: Sequence[str], coord_name: str) -> int | None:
    coord = str(coord_name)
    for idx, label in enumerate(labels):
        lab = str(label)
        if lab == coord:
            return idx
        if lab.endswith(f'/{coord}/value'):
            return idx
        if lab.endswith(f'/{coord}'):
            return idx
    return None


def _table_to_array(table: osim.TimeSeriesTable) -> Tuple[np.ndarray, List[str]]:
    labels = [table.getColumnLabels().get(i) for i in range(table.getNumColumns())]
    times = np.array([float(table.getIndependentColumn().get(i)) for i in range(table.getNumRows())], dtype=np.float64)
    rows: List[List[float]] = []
    for row in range(table.getNumRows()):
        rows.append([float(table.getDependentColumn(i).get(row)) for i in range(table.getNumColumns())])
    data = np.asarray(rows, dtype=np.float64)
    return (np.column_stack([times, data]), ['time'] + labels)


def _process_reference_table(mot_path: Path, *, lowpass_hz: float) -> Tuple[np.ndarray, List[str]]:
    tp = build_moco_states_table_processor(mot_path, lowpass_hz=float(lowpass_hz))
    try:
        table = tp.process()
        return _table_to_array(table)
    except Exception:
        out_sto = mot_path.with_suffix('.reference.sto')
        tp.processAndPrint(str(out_sto))
        return _storage_to_array(osim.Storage(str(out_sto)))


def _interpolate_columns(data: np.ndarray, labels: List[str], frame_times: np.ndarray, coord_names: Sequence[str]) -> np.ndarray:
    if 'time' not in labels:
        raise RuntimeError('Expected time column in coordinate table')
    time_col = labels.index('time')
    src_times = data[:, time_col]
    out = np.full((int(frame_times.shape[0]), len(coord_names)), np.nan, dtype=np.float64)
    for ci, coord in enumerate(coord_names):
        col_idx = _column_index_for_coordinate(labels, coord)
        if col_idx is None:
            continue
        series = data[:, col_idx]
        finite = np.isfinite(src_times) & np.isfinite(series)
        if finite.sum() < 2:
            continue
        out[:, ci] = np.interp(frame_times, src_times[finite], series[finite], left=np.nan, right=np.nan)
    return out


def build_moco_reference_coordinates(q: np.ndarray, *, fps: float, mapping: RajagopalCoordMapping, frame_times: np.ndarray, work_dir: Path, lowpass_hz: float) -> np.ndarray:
    mot_path = work_dir / 'coordinates.mot'
    write_coordinates_mot(q, mot_path, fps=float(fps), mapping=mapping)
    data, labels = _process_reference_table(mot_path, lowpass_hz=float(lowpass_hz))
    return _interpolate_columns(data, labels, frame_times, mapping.opensim_coord_names)


def extract_simulated_coordinates(moco_sol: osim.MocoSolution, *, mapping: RajagopalCoordMapping, frame_times: np.ndarray) -> np.ndarray:
    table = moco_sol.exportToStatesTrajectoryTable()
    data, labels = _table_to_array(table)
    return _interpolate_columns(data, labels, frame_times, mapping.opensim_coord_names)


def calculate_coordinate_tracking_errors(simulated: np.ndarray, reference: np.ndarray, coord_names: Sequence[str]) -> Dict[str, Any]:
    sim = np.asarray(simulated, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if sim.shape != ref.shape:
        raise ValueError(f'simulated/reference shape mismatch: {sim.shape} vs {ref.shape}')
    per_coord: List[Dict[str, Any]] = []
    max_trans_rmse = 0.0
    max_rot_rmse = 0.0
    worst_trans = ''
    worst_rot = ''
    for ci, name in enumerate(coord_names):
        err = sim[:, ci] - ref[:, ci]
        finite = np.isfinite(sim[:, ci]) & np.isfinite(ref[:, ci])
        if finite.sum() == 0:
            rmse = float('nan')
            max_abs = float('nan')
        else:
            e = err[finite]
            rmse = float(np.sqrt(np.mean(e * e)))
            max_abs = float(np.max(np.abs(e)))
        is_trans = is_translational_coordinate(name)
        per_coord.append({'coordinate': str(name), 'rmse': rmse, 'max_abs': max_abs, 'is_translational': bool(is_trans)})
        if not np.isfinite(rmse):
            continue
        if is_trans and rmse > max_trans_rmse:
            max_trans_rmse = rmse
            worst_trans = str(name)
        elif not is_trans and rmse > max_rot_rmse:
            max_rot_rmse = rmse
            worst_rot = str(name)
    return {'per_coordinate': per_coord, 'max_translational_rmse_m': float(max_trans_rmse), 'max_rotational_rmse_deg': float(max_rot_rmse), 'worst_translational_coordinate': worst_trans, 'worst_rotational_coordinate': worst_rot}


def pool_coordinate_tracking_errors(segment_errors: Sequence[Dict[str, Any]], coord_names: Sequence[str]) -> Dict[str, Any]:
    if not segment_errors:
        return calculate_coordinate_tracking_errors(np.zeros((0, len(coord_names))), np.zeros((0, len(coord_names))), coord_names)
    pooled_sim: List[np.ndarray] = []
    pooled_ref: List[np.ndarray] = []
    for item in segment_errors:
        sim = np.asarray(item.get('simulated'), dtype=np.float64)
        ref = np.asarray(item.get('reference'), dtype=np.float64)
        if sim.size == 0 or ref.size == 0 or sim.shape != ref.shape:
            continue
        pooled_sim.append(sim)
        pooled_ref.append(ref)
    if not pooled_sim:
        return calculate_coordinate_tracking_errors(np.zeros((0, len(coord_names))), np.zeros((0, len(coord_names))), coord_names)
    sim_all = np.concatenate(pooled_sim, axis=0)
    ref_all = np.concatenate(pooled_ref, axis=0)
    return calculate_coordinate_tracking_errors(sim_all, ref_all, coord_names)


def summarize_coordinate_tracking_stats(tracking: Dict[str, Any]) -> Dict[str, float | str]:
    out: Dict[str, float | str] = {
        'max_translational_coord_rmse_m': float(tracking.get('max_translational_rmse_m', 0.0)),
        'max_rotational_coord_rmse_deg': float(tracking.get('max_rotational_rmse_deg', 0.0)),
    }
    worst_trans = tracking.get('worst_translational_coordinate')
    worst_rot = tracking.get('worst_rotational_coordinate')
    if worst_trans:
        out['worst_translational_coordinate'] = str(worst_trans)
    if worst_rot:
        out['worst_rotational_coordinate'] = str(worst_rot)
    return out
