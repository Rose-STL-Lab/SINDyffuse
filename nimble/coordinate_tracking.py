from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import opensim as osim
from nimble.muscle_activation import _storage_to_array
from nimble.rajagopal_coord_map import NIMBLE_TO_OPENSIM_COORD, RajagopalCoordMapping, build_moco_states_table_processor, write_coordinates_mot

# Default per-coordinate RMSE thresholds for post-Moco quality gates.
DEFAULT_MAX_TRANSLATIONAL_COORD_RMSE_M = 0.02
DEFAULT_MAX_ROTATIONAL_COORD_RMSE_DEG = 5.0
TRACKED_OPENSIM_COORD_NAMES: Tuple[str, ...] = tuple(NIMBLE_TO_OPENSIM_COORD.values())


def short_coord_name(name: str) -> str:
    """Strip OpenSim path + /value|/speed so pelvis_tilt matches /jointset/.../pelvis_tilt/value."""
    text = str(name).strip()
    parts = [p for p in text.split('/') if p]
    if not parts:
        return text
    last = parts[-1]
    if last in {'value', 'speed', 'u'} and len(parts) >= 2:
        return parts[-2]
    return last


def is_translational_coordinate(name: str) -> bool:
    short = short_coord_name(name)
    return any((short.endswith(suffix) for suffix in ('_tx', '_ty', '_tz')))


def is_tracked_opensim_coordinate(name: str) -> bool:
    return short_coord_name(name) in TRACKED_OPENSIM_COORD_NAMES


def _is_time_label(label: str) -> bool:
    return str(label).strip().lower() == 'time'


def _is_speed_label(label: str) -> bool:
    lab = str(label).strip().lower()
    return lab.endswith('/speed') or lab.endswith('/u') or lab.endswith('_u')


def _column_index_for_coordinate(labels: Sequence[str], coord_name: str) -> int | None:
    coord = short_coord_name(coord_name)
    value_idx: int | None = None
    any_idx: int | None = None
    for idx, label in enumerate(labels):
        if _is_time_label(label):
            continue
        short = short_coord_name(label)
        if short != coord:
            continue
        if _is_speed_label(label):
            if any_idx is None:
                any_idx = idx
            continue
        value_idx = idx
        break
    return value_idx if value_idx is not None else any_idx


def _vector_to_1d(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float64).reshape(-1)
    if hasattr(value, 'nrow') and hasattr(value, 'get'):
        n_row = int(value.nrow())
        n_col = int(value.ncol()) if hasattr(value, 'ncol') else 1
        if n_col <= 1:
            return np.array([float(value.get(i, 0)) for i in range(n_row)], dtype=np.float64)
        return np.array([float(value.get(i, 0)) for i in range(n_row)], dtype=np.float64)
    if hasattr(value, 'size') and hasattr(value, 'get'):
        return np.array([float(value.get(i)) for i in range(int(value.size()))], dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _dependent_column_series(table: Any, *, index: int, label: str, n_rows: int) -> np.ndarray:
    getters = (
        lambda: table.getDependentColumn(str(label)),
        lambda: table.getDependentColumnAtIndex(int(index)),
        lambda: table.getDependentColumn(int(index)),
    )
    last_err: Exception | None = None
    for getter in getters:
        try:
            series = _vector_to_1d(getter())
        except Exception as exc:
            last_err = exc
            continue
        if series.size >= n_rows:
            return np.asarray(series[:n_rows], dtype=np.float64)
        if series.size > 0:
            out = np.full(n_rows, np.nan, dtype=np.float64)
            out[:series.size] = series
            return out
    raise RuntimeError(f'Could not read TimeSeriesTable column {label!r} (index {index}): {last_err}')


def _table_to_array(table: osim.TimeSeriesTable) -> Tuple[np.ndarray, List[str]]:
    n_cols = int(table.getNumColumns())
    n_rows = int(table.getNumRows())
    raw_labels = [str(table.getColumnLabels().get(i)) for i in range(n_cols)]
    start = 0
    if raw_labels and _is_time_label(raw_labels[0]):
        start = 1
        raw_labels = raw_labels[1:]
    times = _vector_to_1d(table.getIndependentColumn())
    if times.size < n_rows:
        raise RuntimeError(f'TimeSeriesTable independent column shorter than rows: {times.size} < {n_rows}')
    times = times[:n_rows]
    columns = [_dependent_column_series(table, index=start + i, label=lab, n_rows=n_rows) for i, lab in enumerate(raw_labels)]
    data = np.column_stack([times] + columns) if columns else times.reshape(-1, 1)
    return (data, ['time'] + raw_labels)


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


def _moco_state_names(moco_sol: Any) -> List[str]:
    for attr in ('getStateNames',):
        fn = getattr(moco_sol, attr, None)
        if fn is None:
            continue
        try:
            names = fn()
        except TypeError:
            continue
        try:
            return [str(names.get(i)) for i in range(int(names.size()))]
        except Exception:
            try:
                return [str(n) for n in list(names)]
            except Exception:
                continue
    return []


def _moco_state_times(moco_sol: Any) -> np.ndarray:
    for attr in ('getTime', 'getTimeMat'):
        fn = getattr(moco_sol, attr, None)
        if fn is None:
            continue
        try:
            return _vector_to_1d(fn())
        except TypeError:
            continue
    table = moco_sol.exportToStatesTrajectoryTable()
    return _vector_to_1d(table.getIndependentColumn())


def _moco_state_series(moco_sol: Any, name: str) -> np.ndarray:
    get_mat = getattr(moco_sol, 'getStateMat', None)
    if get_mat is not None:
        try:
            return _vector_to_1d(get_mat(name))
        except TypeError:
            pass
    get_state = getattr(moco_sol, 'getState', None)
    if get_state is not None:
        return _vector_to_1d(get_state(name))
    raise RuntimeError(f'OpenSim MocoSolution has no per-state accessor for {name!r}')


def _extract_simulated_from_state_mats(moco_sol: Any, coord_names: Sequence[str], frame_times: np.ndarray) -> np.ndarray | None:
    if getattr(moco_sol, 'getStateMat', None) is None and getattr(moco_sol, 'getState', None) is None:
        return None
    try:
        state_names = _moco_state_names(moco_sol)
        sol_times = _moco_state_times(moco_sol)
    except Exception:
        return None
    if sol_times.size < 2:
        return None
    out = np.full((int(frame_times.shape[0]), len(coord_names)), np.nan, dtype=np.float64)
    matched = 0
    for ci, coord in enumerate(coord_names):
        name = None
        if state_names:
            idx = _column_index_for_coordinate(state_names, coord)
            if idx is not None:
                name = state_names[idx]
        if name is None:
            continue
        try:
            series = _moco_state_series(moco_sol, name)
        except Exception:
            continue
        n = min(series.size, sol_times.size)
        if n < 2:
            continue
        finite = np.isfinite(sol_times[:n]) & np.isfinite(series[:n])
        if finite.sum() < 2:
            continue
        out[:, ci] = np.interp(frame_times, sol_times[:n][finite], series[:n][finite], left=np.nan, right=np.nan)
        matched += 1
    return out if matched else None


def extract_simulated_coordinates(moco_sol: osim.MocoSolution, *, mapping: RajagopalCoordMapping, frame_times: np.ndarray) -> np.ndarray:
    table_out: np.ndarray | None = None
    table_err: Exception | None = None
    try:
        table = moco_sol.exportToStatesTrajectoryTable()
        data, labels = _table_to_array(table)
        table_out = _interpolate_columns(data, labels, frame_times, mapping.opensim_coord_names)
    except Exception as exc:
        table_err = exc
    mat_out = _extract_simulated_from_state_mats(moco_sol, mapping.opensim_coord_names, frame_times)
    if mat_out is None:
        if table_out is None:
            raise RuntimeError(f'failed to extract simulated coordinates: {table_err}') from table_err
        return table_out
    if table_out is not None:
        missing = ~np.isfinite(mat_out).any(axis=0)
        if missing.any():
            mat_out[:, missing] = table_out[:, missing]
    return mat_out


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
