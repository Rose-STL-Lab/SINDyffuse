from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
from common.paths import default_humanml3d_root, resolve_repo_path
from nimble.muscle_activation import MuscleActivationConfig

def _ik_mean_fk_loss(ik_stats: Dict[str, Any]) -> float | None:
    val = ik_stats.get('mean_fk_loss')
    if val is None:
        val = ik_stats.get('mean_fit_joints_loss', ik_stats.get('mean_ik_error'))
    if val is None:
        return None
    out = float(val)
    return out if np.isfinite(out) else None

def _ik_max_fk_loss(ik_stats: Dict[str, Any]) -> float | None:
    val = ik_stats.get('max_fk_loss')
    if val is None:
        per = ik_stats.get('per_frame_fk_loss')
        if isinstance(per, (list, tuple)) and per:
            finite = [float(x) for x in per if np.isfinite(float(x))]
            if finite:
                return float(max(finite))
        return None
    out = float(val)
    return out if np.isfinite(out) else None

@dataclass
class IkGateConfig:
    max_mean_fk_loss: float | None = None
    max_max_fk_loss: float | None = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IkGateConfig':
        mean = data.get('max_mean_fk_loss')
        mx = data.get('max_max_fk_loss')
        return cls(max_mean_fk_loss=float(mean) if mean is not None and float(mean) > 0.0 else None, max_max_fk_loss=float(mx) if mx is not None and float(mx) > 0.0 else None)

    def merge_cli(self, cfg: MuscleActivationConfig) -> 'IkGateConfig':
        mean = cfg.moco_max_mean_fk_loss if cfg.moco_max_mean_fk_loss is not None and float(cfg.moco_max_mean_fk_loss) > 0.0 else self.max_mean_fk_loss
        mx = cfg.moco_max_max_fk_loss if cfg.moco_max_max_fk_loss is not None and float(cfg.moco_max_max_fk_loss) > 0.0 else self.max_max_fk_loss
        return IkGateConfig(max_mean_fk_loss=mean, max_max_fk_loss=mx)

def default_ik_gate_config_path(out_root: str | Path | None=None) -> Path:
    root = Path(out_root).expanduser().resolve() if out_root else Path(default_humanml3d_root())
    return root / 'ik_gate_config.json'

def load_ik_gate_config(path: str | Path | None=None, *, out_root: str | Path | None=None) -> IkGateConfig:
    candidates: list[Path] = []
    if path:
        candidates.append(resolve_repo_path(str(path)))
    candidates.append(default_ik_gate_config_path(out_root))
    for candidate in candidates:
        if candidate.is_file():
            data = json.loads(candidate.read_text(encoding='utf-8'))
            return IkGateConfig.from_dict(data)
    return IkGateConfig()

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

def evaluate_ik_fk_gate(ik_stats: Dict[str, Any], *, gate_cfg: IkGateConfig) -> Tuple[bool, str]:
    if gate_cfg.max_mean_fk_loss is not None:
        loss = _ik_mean_fk_loss(ik_stats)
        if loss is not None and loss > float(gate_cfg.max_mean_fk_loss):
            return (False, f'mean_fk_loss {loss:.6g} > {gate_cfg.max_mean_fk_loss:.6g}')
    if gate_cfg.max_max_fk_loss is not None:
        loss = _ik_max_fk_loss(ik_stats)
        if loss is not None and loss > float(gate_cfg.max_max_fk_loss):
            return (False, f'max_fk_loss {loss:.6g} > {gate_cfg.max_max_fk_loss:.6g}')
    return (True, '')

def evaluate_ik_gate(ik_stats: Dict[str, Any], *, q: np.ndarray | None=None, gate_cfg: IkGateConfig) -> Tuple[bool, str]:
    if q is not None:
        ok, reason = q_trajectory_is_valid(q)
        if not ok:
            return (False, reason)
    return evaluate_ik_fk_gate(ik_stats, gate_cfg=gate_cfg)

def evaluate_moco_preflight_gate(*, ik_manifest_status: str | None, q: np.ndarray, ik_stats: Dict[str, Any], gate_cfg: IkGateConfig) -> Tuple[bool, str]:
    if ik_manifest_status in {'ik_failed', 'error'}:
        return (False, f'prior IK status {ik_manifest_status}')
    ok, reason = evaluate_ik_gate(ik_stats, q=q, gate_cfg=gate_cfg)
    if not ok:
        return (False, reason)
    return (True, '')

def evaluate_activation_gate(ik_stats: Dict[str, Any], *, num_frames: int, cfg: MuscleActivationConfig, gate_cfg: IkGateConfig | None=None) -> Tuple[bool, str]:
    del num_frames
    merged = (gate_cfg or IkGateConfig()).merge_cli(cfg)
    return evaluate_ik_fk_gate(ik_stats, gate_cfg=merged)

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
    return out

def derive_moco_manifest_status(*, segment_success_count: int, moco_skipped: bool=False) -> str:
    if moco_skipped:
        return 'moco_skipped'
    if int(segment_success_count) <= 0:
        return 'moco_failed'
    return 'ok'
