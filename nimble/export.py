from __future__ import annotations
import time
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
from nimble.activation_gates import CoordinateTrackingGateConfig, IkGateConfig, activation_valid_fraction, derive_moco_manifest_status, evaluate_coordinate_tracking_gate, evaluate_ik_gate, evaluate_moco_preflight_gate, summarize_moco_metadata, summarize_moco_segment_failures
from nimble.coordinate_tracking import summarize_coordinate_tracking_stats
from common.run_logging import append_verbose_log
from nimble.muscle_activation import MuscleActivationConfig, MuscleActivationResult, compute_muscle_activation, configure_opensim_logging, muscle_names, normalize_activation_method
from nimble.opensim_log import opensim_quiet
from nimble.opensim_ik import clear_opensim_ik_cache, fit_q_lai
from nimble.lai_coord_map import LAI_NUM_DOFS
from nimble.lai_features import features_from_lai_q, zero_bio_matrix
from nimble.moco_segment import SIM_GRF_COLS
from nimble.muscle_b3d import MUSCLE_ACTIVATION_ROWS
from datasets.lai_cache import has_muscle_activations, read_motion_npz, write_motion_npz


def clear_export_caches() -> None:
    import gc
    clear_opensim_ik_cache()
    gc.collect()


def _stats_dict(ik_stats: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, str]]:
    stats: Dict[str, float] = {}
    meta_strings: Dict[str, str] = {}
    for k, v in ik_stats.items():
        if isinstance(v, str):
            meta_strings[k] = v
        elif isinstance(v, (int, float, np.integer, np.floating)):
            stats[k] = float(v)
        elif isinstance(v, bool):
            stats[k] = float(v)
    return (stats, meta_strings)


def _write_npz_from_lai_q(
    *,
    q_lai: np.ndarray,
    output_npz_path: str | Path,
    trial_name: str,
    fps: float,
    mass_kg: float,
    height_m: float,
    muscle_act: np.ndarray,
    sim_grf: np.ndarray | None,
    activation_mask: np.ndarray | None,
    ik_stats: Dict[str, Any],
    activation_diagnostics: Dict[str, Any] | None=None,
) -> Tuple[Dict[str, float], int, Dict[str, str]]:
    """Persist Lai q [T, 31] or [31, T] plus optional activation channels to NPZ."""
    arr = np.asarray(q_lai, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f'Expected Lai q 2D, got {arr.shape}')
    if arr.shape[0] == LAI_NUM_DOFS:
        q_tq = arr.T
    elif arr.shape[1] == LAI_NUM_DOFS:
        q_tq = arr
    else:
        raise ValueError(f'Expected Lai q with {LAI_NUM_DOFS} DOFs, got {arr.shape}')
    t_len = int(q_tq.shape[0])
    u, c, _, _ = features_from_lai_q(np.asarray(q_tq, dtype=np.float64), fps=float(fps))
    bio = zero_bio_matrix(t_len)
    write_motion_npz(
        output_npz_path,
        motion_id=str(trial_name),
        q=q_tq,
        fps=float(fps),
        mass_kg=float(mass_kg),
        height_m=float(height_m),
        muscle_activations=muscle_act,
        muscle_activation_mask=activation_mask,
        sim_grf=sim_grf,
        sindy_u=u,
        sindy_c=c,
        guidance_bio=bio,
        extra={'activation_diagnostics_json': np.asarray(json.dumps(activation_diagnostics))} if activation_diagnostics is not None else None,
    )
    ik_stats['guidance_features_computed'] = 0.0
    ik_stats['sindy_features_computed'] = 1.0
    ik_stats['opensim_model'] = 'LaiUhlrich2022'
    ik_stats['lai_num_dofs'] = float(LAI_NUM_DOFS)
    stats, meta_strings = _stats_dict(ik_stats)
    return (stats, int(LAI_NUM_DOFS), meta_strings)


def export_ik_to_npz(
    hml3d_positions: np.ndarray,
    output_npz_path: str | Path,
    *,
    trial_name: str,
    fps: float = 20.0,
    mass_kg: float = 70.0,
    height_m: float = 1.75,
    gate_cfg: IkGateConfig | None = None,
    opensim_log_level: str = 'Off',
) -> Tuple[Dict[str, float], int, Dict[str, str], str]:
    poses = np.asarray(hml3d_positions, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (22, 3):
        raise ValueError(f'Expected hml3d_positions [T,22,3], got {poses.shape}')
    configure_opensim_logging(opensim_log_level)
    num_input_frames = int(poses.shape[0])
    append_verbose_log(f'{trial_name}: Lai OpenSim IK start ({num_input_frames} frames)')
    with opensim_quiet(opensim_log_level):
        poses_q, ik_stats = fit_q_lai(poses)
    del poses
    per_frame_fk = ik_stats.pop('per_frame_fk_loss', None)
    per_frame_solver = ik_stats.pop('per_frame_loss', None)
    append_verbose_log(
        f"{trial_name}: IK fitting done mean_fk_loss={ik_stats.get('mean_fk_loss', float('nan')):.6f} "
        f"mean_ik_error={ik_stats.get('mean_ik_error', float('nan')):.6f} "
        f"success_ratio={ik_stats.get('success_ratio', 0.0):.4f} frames={int(ik_stats.get('total_frames', 0))}"
    )
    if per_frame_fk is not None:
        ik_stats['per_frame_fk_loss'] = per_frame_fk
    if per_frame_solver is not None:
        ik_stats['per_frame_loss'] = per_frame_solver
    ik_stats['pose_smoothing_enabled'] = 0.0
    ik_stats['activation_method'] = 'ik'
    # poses_q is [31, T]
    q_tq = np.ascontiguousarray(poses_q.T, dtype=np.float64)
    del poses_q
    merged_gate = gate_cfg or IkGateConfig.default()
    ik_ok, ik_reason = evaluate_ik_gate(ik_stats, q=q_tq, gate_cfg=merged_gate)
    manifest_status = 'ik_ok' if ik_ok else 'ik_failed'
    if not ik_ok:
        ik_stats['ik_gate_reason'] = ik_reason
    else:
        ik_stats.pop('ik_gate_reason', None)
    num_frames = int(q_tq.shape[0])
    muscle_act = np.zeros((num_frames, MUSCLE_ACTIVATION_ROWS), dtype=np.float32)
    ik_stats['muscle_activation_skipped'] = 1.0
    ik_stats['muscle_activation_computed'] = 0.0
    stats, num_dofs, meta_strings = _write_npz_from_lai_q(
        q_lai=q_tq.astype(np.float32),
        output_npz_path=output_npz_path,
        trial_name=trial_name,
        fps=fps,
        mass_kg=mass_kg,
        height_m=height_m,
        muscle_act=muscle_act,
        sim_grf=None,
        activation_mask=np.zeros(num_frames, dtype=np.float32),
        ik_stats=ik_stats,
    )
    clear_export_caches()
    return (stats, num_dofs, meta_strings, manifest_status)


# Backward-compatible aliases
export_ik_to_b3d = export_ik_to_npz


def patch_npz_activations(
    npz_path: str | Path,
    *,
    trial_name: str,
    act_cfg: MuscleActivationConfig,
    ik_manifest_status: str | None = None,
    ik_stats: Dict[str, Any] | None = None,
    tracking_gate_cfg: CoordinateTrackingGateConfig | None = None,
) -> Tuple[Dict[str, float], int, Dict[str, str], str]:
    configure_opensim_logging(act_cfg.opensim_log_level)
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing Lai cache NPZ: {path}')
    cached = read_motion_npz(path)
    q_lai = np.asarray(cached['q'], dtype=np.float64)
    num_frames = int(q_lai.shape[0])
    mass_kg = float(cached['mass_kg'])
    height_m = float(cached['height_m'])
    stats: Dict[str, Any] = dict(ik_stats or {})
    allowed, reason = evaluate_moco_preflight_gate(ik_manifest_status=ik_manifest_status, q=q_lai)
    method = normalize_activation_method(act_cfg.activation_method)
    stats['activation_method'] = method
    sim_grf: np.ndarray | None = None
    activation_mask: np.ndarray | None = None
    activation_diagnostics = None
    muscle_act = np.full((num_frames, MUSCLE_ACTIVATION_ROWS), np.nan, dtype=np.float32)
    if not allowed:
        stats['moco_skipped_reason'] = reason
        stats['muscle_activation_skipped'] = 1.0
        stats['muscle_activation_computed'] = 0.0
        activation_mask = np.zeros(num_frames, dtype=np.float32)
        manifest_status = 'moco_skipped'
    else:
        append_verbose_log(f'{trial_name}: {method} start ({num_frames} frames, mesh={act_cfg.mesh_interval})')
        t0 = time.perf_counter()
        try:
            act_result = compute_muscle_activation(q_lai, cfg=act_cfg)
        except Exception as exc:
            n_muscles = len(muscle_names())
            act_result = MuscleActivationResult(
                activations=np.full((num_frames, n_muscles), np.nan, dtype=np.float32),
                muscle_names=muscle_names(),
                metadata={
                    'activation_method': method,
                    'activation_validity_mask': np.zeros(num_frames, dtype=np.float32),
                    'error': str(exc),
                    'moco_segment_success_count': 0,
                },
                forces=np.full((num_frames, SIM_GRF_COLS), np.nan, dtype=np.float32),
            )
        stats['muscle_activation_seconds'] = float(time.perf_counter() - t0)
        stats['muscle_activation_computed'] = 1.0
        seg_count = act_result.metadata.get('moco_segment_success_count', 0)
        append_verbose_log(
            f"{trial_name}: {method} done seconds={stats['muscle_activation_seconds']:.1f} "
            f"segment_success={seg_count}/{act_result.metadata.get('moco_segment_count', 'n/a')}"
        )
        for key, val in summarize_moco_metadata(act_result.metadata).items():
            if isinstance(val, (int, float)):
                stats[f'moco_{key}'] = float(val)
        if act_result.metadata.get('moco_segment_count') is not None:
            stats['moco_segment_count'] = float(act_result.metadata['moco_segment_count'])
        if act_result.metadata.get('moco_segment_success_count') is not None:
            stats['moco_segment_success_count'] = float(act_result.metadata['moco_segment_success_count'])
        if act_result.metadata.get('ground_offset_m') is not None:
            stats['moco_ground_offset_m'] = float(act_result.metadata['ground_offset_m'])
        muscle_act = np.asarray(act_result.activations, dtype=np.float32)
        if act_result.forces is not None:
            sim_grf = np.asarray(act_result.forces, dtype=np.float32)
        elif act_result.metadata.get('sim_grf') is not None:
            sim_grf = np.asarray(act_result.metadata['sim_grf'], dtype=np.float32)
        mask = act_result.metadata.get('activation_validity_mask')
        activation_mask = (
            np.asarray(mask, dtype=np.float32).reshape(-1)
            if mask is not None
            else np.isfinite(muscle_act).all(axis=1).astype(np.float32)
        )
        stats['activation_valid_fraction'] = activation_valid_fraction(muscle_act, activation_mask)
        stats['moco_segment_success_fraction'] = float(act_result.metadata.get('moco_segment_success_fraction', 0.0))
        tracking_metrics = act_result.metadata.get('coordinate_tracking') or {}
        summary = summarize_coordinate_tracking_stats(tracking_metrics) if tracking_metrics.get('per_coordinate') else {}
        for key, val in summary.items():
            stats[str(key)] = val
        activation_diagnostics = {key: act_result.metadata.get(key) for key in
            ('label_processing_version', 'coordinate_tracking', 'moco_segment_details', 'grf_torque_convention')}
        seg_success = int(act_result.metadata.get('moco_segment_success_count', 0))
        tracking_ok = True
        if seg_success <= 0:
            fail_reason = summarize_moco_segment_failures(act_result.metadata)
            stats['moco_failed_reason'] = fail_reason
            tracking_ok = False
        else:
            tracking_ok, tracking_reason = evaluate_coordinate_tracking_gate(
                act_result.metadata.get('coordinate_tracking'), gate_cfg=tracking_gate_cfg
            )
            if not tracking_ok:
                stats['coordinate_tracking_gate_reason'] = tracking_reason
                tracking_ok = True
        manifest_status = derive_moco_manifest_status(segment_success_count=seg_success, tracking_ok=tracking_ok)
        del act_result
    out_stats, num_dofs, meta_strings = _write_npz_from_lai_q(
        q_lai=q_lai.astype(np.float32),
        output_npz_path=path,
        trial_name=trial_name,
        fps=float(act_cfg.fps),
        mass_kg=mass_kg,
        height_m=height_m,
        muscle_act=muscle_act,
        sim_grf=sim_grf,
        activation_mask=activation_mask,
        ik_stats=stats,
        activation_diagnostics=activation_diagnostics,
    )
    clear_export_caches()
    return (out_stats, num_dofs, meta_strings, manifest_status)


patch_b3d_moco = patch_npz_activations


def export_motion_to_npz(
    hml3d_positions: np.ndarray,
    output_npz_path: str | Path,
    *,
    trial_name: str,
    fps: float = 20.0,
    mass_kg: float = 70.0,
    height_m: float = 1.75,
    muscle_activation_cfg: MuscleActivationConfig | None = None,
    skip_muscle_activation: bool = False,
    activation_method: str | None = None,
    gate_cfg: IkGateConfig | None = None,
) -> Tuple[Dict[str, float], int, Dict[str, str]]:
    act_cfg = replace(
        muscle_activation_cfg or MuscleActivationConfig(fps=float(fps), mass_kg=float(mass_kg)),
        fps=float(fps),
        mass_kg=float(mass_kg),
    )
    if activation_method is not None:
        act_cfg = replace(act_cfg, activation_method=normalize_activation_method(activation_method))
    ik_stats, num_dofs, meta_strings, ik_status = export_ik_to_npz(
        hml3d_positions,
        output_npz_path,
        trial_name=trial_name,
        fps=fps,
        mass_kg=mass_kg,
        height_m=height_m,
        gate_cfg=gate_cfg,
        opensim_log_level=act_cfg.opensim_log_level,
    )
    if skip_muscle_activation:
        return (ik_stats, num_dofs, meta_strings)
    moco_stats, num_dofs, meta_strings, _ = patch_npz_activations(
        output_npz_path, trial_name=trial_name, act_cfg=act_cfg, ik_manifest_status=ik_status, ik_stats=ik_stats
    )
    return (moco_stats, num_dofs, meta_strings)


export_motion_to_b3d = export_motion_to_npz


def npz_has_activations(path: str | Path) -> bool:
    return has_muscle_activations(path)
