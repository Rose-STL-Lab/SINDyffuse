from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
from nimble.coordinate_tracking import DEFAULT_MAX_ROTATIONAL_COORD_RMSE_DEG, DEFAULT_MAX_TRANSLATIONAL_COORD_RMSE_M, is_tracked_opensim_coordinate, is_translational_coordinate, short_coord_name
from nimble.muscle_activation import MuscleActivationConfig

IK_MIN_SUCCESS_RATIO = 1.0

@dataclass
class CoordinateTrackingGateConfig:
    max_translational_rmse_m: float = DEFAULT_MAX_TRANSLATIONAL_COORD_RMSE_M
    max_rotational_rmse_deg: float = DEFAULT_MAX_ROTATIONAL_COORD_RMSE_DEG

    @classmethod
    def default(cls) -> 'CoordinateTrackingGateConfig':
        return cls()

@dataclass
class IkGateConfig:
    min_success_ratio: float = IK_MIN_SUCCESS_RATIO

    @classmethod
    def default(cls) -> 'IkGateConfig':
        return cls()

def _ik_success_ratio(ik_stats: Dict[str, Any]) -> float | None:
    val = ik_stats.get('success_ratio')
    if val is None:
        return None
    out = float(val)
    return out if np.isfinite(out) else None

def q_trajectory_is_valid(q: np.ndarray) -> Tuple[bool, str]:
    arr = np.asarray(q, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 1:
        return (False, f'invalid q shape {arr.shape}')
    if not np.all(np.isfinite(arr)):
        return (False, 'non-finite q values')
    if arr.shape[0] < 2:
        return (False, 'degenerate q (fewer than 2 frames)')
    row_norms = np.linalg.norm(arr, axis=1)
    if not np.any(row_norms > 1e-08):
        return (False, 'all q frames degenerate')
    return (True, '')

def evaluate_ik_solver_gate(ik_stats: Dict[str, Any], *, gate_cfg: IkGateConfig) -> Tuple[bool, str]:
    ratio = _ik_success_ratio(ik_stats)
    if ratio is not None and ratio + 1e-12 < float(gate_cfg.min_success_ratio):
        return (False, f'success_ratio {ratio:.4f} < {gate_cfg.min_success_ratio:.4f}')
    return (True, '')

def evaluate_ik_gate(ik_stats: Dict[str, Any], *, q: np.ndarray | None=None, gate_cfg: IkGateConfig | None=None) -> Tuple[bool, str]:
    merged = gate_cfg or IkGateConfig.default()
    if q is not None:
        ok, reason = q_trajectory_is_valid(q)
        if not ok:
            return (False, reason)
    return evaluate_ik_solver_gate(ik_stats, gate_cfg=merged)

def evaluate_coordinate_tracking_gate(tracking: Dict[str, Any] | None, *, gate_cfg: CoordinateTrackingGateConfig | None=None) -> Tuple[bool, str]:
    merged = gate_cfg or CoordinateTrackingGateConfig.default()
    if not tracking:
        return (False, 'missing coordinate tracking metrics')
    per_coord = tracking.get('per_coordinate') or []
    if not per_coord:
        return (False, 'empty coordinate tracking metrics')
    max_trans = float(tracking.get('max_translational_rmse_m', 0.0))
    max_rot = float(tracking.get('max_rotational_rmse_deg', 0.0))
    missing_tracked: List[str] = []
    gated_any = False
    for row in per_coord:
        name = str(row.get('coordinate', ''))
        rmse = float(row.get('rmse', float('nan')))
        tracked = is_tracked_opensim_coordinate(name)
        if not np.isfinite(rmse):
            if tracked:
                missing_tracked.append(short_coord_name(name) or name or 'unknown coordinate')
            continue
        gated_any = True
        if is_translational_coordinate(name):
            if rmse > float(merged.max_translational_rmse_m):
                return (False, f'translational RMSE {name} {rmse:.6g} m > {merged.max_translational_rmse_m:.6g} m')
        elif rmse > float(merged.max_rotational_rmse_deg):
            return (False, f'rotational RMSE {name} {rmse:.6g} deg > {merged.max_rotational_rmse_deg:.6g} deg')
    if missing_tracked:
        shown = ', '.join(missing_tracked[:8])
        extra = f' (+{len(missing_tracked) - 8} more)' if len(missing_tracked) > 8 else ''
        return (False, f'missing tracking RMSE for {shown}{extra}')
    if not gated_any:
        return (False, 'no finite tracking RMSE for any coordinate')
    if max_trans > float(merged.max_translational_rmse_m):
        worst = str(tracking.get('worst_translational_coordinate', ''))
        return (False, f'max translational RMSE {max_trans:.6g} m > {merged.max_translational_rmse_m:.6g} m ({worst})')
    if max_rot > float(merged.max_rotational_rmse_deg):
        worst = str(tracking.get('worst_rotational_coordinate', ''))
        return (False, f'max rotational RMSE {max_rot:.6g} deg > {merged.max_rotational_rmse_deg:.6g} deg ({worst})')
    return (True, '')

def evaluate_moco_preflight_gate(*, ik_manifest_status: str | None, q: np.ndarray) -> Tuple[bool, str]:
    if ik_manifest_status in {'ik_failed', 'error'}:
        return (False, f'prior IK status {ik_manifest_status}')
    return q_trajectory_is_valid(q)

def evaluate_activation_gate(ik_stats: Dict[str, Any], *, num_frames: int, cfg: MuscleActivationConfig, gate_cfg: IkGateConfig | None=None) -> Tuple[bool, str]:
    del num_frames, cfg
    return evaluate_ik_solver_gate(ik_stats, gate_cfg=gate_cfg or IkGateConfig.default())

def activation_valid_fraction(activations: np.ndarray, mask: np.ndarray | None) -> float:
    act = np.asarray(activations, dtype=np.float64)
    if act.ndim != 2 or act.shape[0] == 0:
        return 0.0
    if mask is not None:
        m = np.asarray(mask, dtype=np.float64).reshape(-1)
        if m.shape[0] == act.shape[0]:
            return float(np.mean(m > 0.5))
    finite_rows = np.isfinite(act).all(axis=1)
    return float(np.mean(finite_rows))

def summarize_moco_metadata(metadata: Dict[str, Any]) -> Dict[str, float | str | int]:
    out: Dict[str, float | str | int] = {'repaired_frame_count': int(metadata.get('repaired_frame_count', 0))}
    obj = metadata.get('moco_objective')
    if obj is not None and np.isfinite(float(obj)):
        out['moco_objective'] = float(obj)
    solve_details = metadata.get('moco_solve_details')
    if isinstance(solve_details, dict):
        if solve_details.get('solver_status'):
            out['moco_solver_status'] = str(solve_details['solver_status'])
        if solve_details.get('solver_success') is not None:
            out['moco_solver_success'] = int(bool(solve_details['solver_success']))
    elif metadata.get('moco_solver_status'):
        out['moco_solver_status'] = str(metadata['moco_solver_status'])
    if metadata.get('moco_solver_success') is not None and 'moco_solver_success' not in out:
        out['moco_solver_success'] = int(bool(metadata['moco_solver_success']))
    details = metadata.get('moco_segment_details') or []
    if details and 'moco_solver_status' not in out:
        statuses = [str(d.get('solver_status', '')) for d in details if isinstance(d, dict) and d.get('solver_status')]
        if statuses:
            out['moco_solver_status'] = statuses[-1]
    if details and 'moco_solver_success' not in out:
        solver_ok = any((isinstance(d, dict) and d.get('solver_success') for d in details))
        out['moco_solver_success'] = int(solver_ok)
    if metadata.get('moco_segment_success_count') is not None:
        out['segment_success_count'] = int(metadata['moco_segment_success_count'])
    if metadata.get('moco_segment_count') is not None:
        out['segment_count'] = int(metadata['moco_segment_count'])
    if metadata.get('max_translational_coord_rmse_m') is not None:
        out['max_translational_coord_rmse_m'] = float(metadata['max_translational_coord_rmse_m'])
    if metadata.get('max_rotational_coord_rmse_deg') is not None:
        out['max_rotational_coord_rmse_deg'] = float(metadata['max_rotational_coord_rmse_deg'])
    return out

def manifest_gate_reason(row: Dict[str, Any]) -> str:
    meta = row.get('meta') or {}
    for key in ('moco_skipped_reason', 'moco_failed_reason', 'coordinate_tracking_gate_reason', 'ik_gate_reason', 'error'):
        val = row.get(key) or meta.get(key)
        if val:
            return str(val)
    return 'failed'

def derive_moco_manifest_status(*, segment_success_count: int, moco_skipped: bool=False, tracking_ok: bool=True) -> str:
    if moco_skipped:
        return 'moco_skipped'
    if int(segment_success_count) <= 0:
        return 'moco_failed'
    if not tracking_ok:
        return 'moco_failed'
    return 'ok'

def summarize_moco_segment_failures(metadata: Dict[str, Any]) -> str:
    """Human-readable reason when every Moco segment fails (avoids empty-tracking false alarms)."""
    details = metadata.get('moco_segment_details') or []
    for key in ('error', 'solver_status', 'coordinate_tracking_error'):
        for detail in details:
            if not isinstance(detail, dict):
                continue
            val = detail.get(key)
            if val:
                return str(val)
    if metadata.get('error'):
        return str(metadata['error'])
    seg_n = int(metadata.get('moco_segment_count') or len(details) or 0)
    return f'all {seg_n} Moco segments failed'
