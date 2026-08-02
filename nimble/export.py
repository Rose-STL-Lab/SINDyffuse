from __future__ import annotations
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import nimblephysics as nimble
from nimble.b3d_schema import B3D_CUSTOM_VALUE_NAMES, pack_b3d_trial_custom_values
from nimble.muscle_b3d import MUSCLE_ACTIVATION_ROWS
from nimble.activation_gates import IkGateConfig, activation_valid_fraction, derive_moco_manifest_status, evaluate_ik_gate, evaluate_moco_preflight_gate, summarize_moco_metadata
from common.run_logging import append_verbose_log
from nimble.muscle_activation import MuscleActivationConfig, MuscleActivationResult, compute_muscle_activation, configure_opensim_logging, muscle_names, normalize_activation_method, opensim_quiet
from nimble.ik import clear_body_ik_cache, fit_q
from nimble.moco_segment import SIM_GRF_COLS
from nimble.physics import load_model
from nimble.skeleton_registry import get_spec
from datasets.nimble_dataset import read_q_frames
from datasets.splits import kinematics_pass_index
from sindy.features import features_from_q
from sindy.targets import bio_matrix, default_physics_cfg
_SKELETON_CACHE: Dict[str, Any] = {}

def clear_export_caches() -> None:
    import gc
    _SKELETON_CACHE.clear()
    clear_body_ik_cache()
    from nimble.b3d_schema import clear_b3d_schema_caches
    from nimble.physics import clear_cache
    clear_b3d_schema_caches()
    clear_cache()
    gc.collect()

def _get_skeleton(*, opensim_log_level: str='Off') -> Tuple[Any, Any]:
    sk = _SKELETON_CACHE.get('rajagopal')
    if sk is None:
        with opensim_quiet(opensim_log_level):
            sk = load_model().skeleton
        _SKELETON_CACHE['rajagopal'] = sk
    return (sk, get_spec('rajagopal'))

def _populate_pass_derived_values(kin_pass: Any, skeleton: Any, poses_q: np.ndarray, dt: float, foot_body_names: Tuple[str, str], *, root_history_len: int=10, root_history_stride: int=3) -> None:
    poses = np.ascontiguousarray(poses_q, dtype=np.float64)
    kin_pass.computeValuesFromForcePlates(skeleton, float(dt), poses, list(foot_body_names), [], rootHistoryLen=int(root_history_len), rootHistoryStride=int(root_history_stride))

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

def _write_b3d_from_q(*, poses_q_f32: np.ndarray, output_b3d_path: str | Path, trial_name: str, fps: float, mass_kg: float, height_m: float, sk: Any, spec: Any, muscle_act: np.ndarray, sim_grf_pack: np.ndarray | None, activation_mask_pack: np.ndarray | None, ik_stats: Dict[str, Any]) -> Tuple[Dict[str, float], int, Dict[str, str]]:
    foot_body_names = tuple(spec.foot_body_names)
    dt = 1.0 / max(float(fps), 1e-06)
    num_dofs = int(sk.getNumDofs())
    num_frames = int(poses_q_f32.shape[1])
    q_traj = np.ascontiguousarray(poses_q_f32.T, dtype=np.float64)
    b3d_subject = nimble.biomechanics.SubjectOnDiskHeader()
    b3d_subject_pass = b3d_subject.addProcessingPass()
    b3d_subject_pass.setProcessingPassType(nimble.biomechanics.ProcessingPassType.KINEMATICS)
    b3d_subject.setHeightM(float(height_m))
    b3d_subject.setMassKg(float(mass_kg))
    b3d_subject.setNumDofs(num_dofs)
    b3d_subject.setNumJoints(int(sk.getNumJoints()))
    b3d_subject.setGroundForceBodies(list(foot_body_names))
    b3d_subject.setNotes(f'Converted from HumanML3D motion: {trial_name}')
    b3d_trial = b3d_subject.addTrial()
    b3d_trial.setName(str(trial_name))
    b3d_trial.setTrialLength(int(poses_q_f32.shape[1]))
    b3d_trial.setTimestep(float(dt))
    b3d_trial.setForcePlates([])
    kin_pass = b3d_trial.addPass()
    kin_pass.setType(nimble.biomechanics.ProcessingPassType.KINEMATICS)
    kin_pass.setPoses(poses_q_f32)
    b3d_trial.setMarkerObservations([{} for _ in range(num_frames)])
    _populate_pass_derived_values(kin_pass, sk, poses_q_f32, dt, foot_body_names, root_history_len=10, root_history_stride=3)
    bio_cfg = default_physics_cfg(fps=float(fps), max_frames=num_frames)
    bio = bio_matrix(q_traj, fps=float(fps), guidance_cfg=bio_cfg)
    u, c, _, _ = features_from_q(q_traj, sk, fps=float(fps))
    b3d_subject.setCustomValueNames(list(B3D_CUSTOM_VALUE_NAMES))
    b3d_trial.setCustomValues(pack_b3d_trial_custom_values(muscle_activations=muscle_act, guidance_bio=bio, sindy_u=u, sindy_c=c, sim_grf=sim_grf_pack, muscle_activation_mask=activation_mask_pack))
    ik_stats['guidance_features_computed'] = 1.0
    ik_stats['sindy_features_computed'] = 1.0
    out_path = Path(output_b3d_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nimble.biomechanics.SubjectOnDisk.writeB3D(str(out_path), b3d_subject)
    del b3d_subject, bio, u, c, muscle_act, q_traj
    stats, meta_strings = _stats_dict(ik_stats)
    return (stats, num_dofs, meta_strings)

def export_ik_to_b3d(hml3d_positions: np.ndarray, output_b3d_path: str | Path, *, trial_name: str, fps: float=20.0, mass_kg: float=70.0, height_m: float=1.75, gate_cfg: IkGateConfig | None=None, opensim_log_level: str='Off') -> Tuple[Dict[str, float], int, Dict[str, str], str]:
    poses = np.asarray(hml3d_positions, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (22, 3):
        raise ValueError(f'Expected hml3d_positions [T,22,3], got {poses.shape}')
    configure_opensim_logging(opensim_log_level)
    sk, spec = _get_skeleton(opensim_log_level=opensim_log_level)
    num_input_frames = int(poses.shape[0])
    append_verbose_log(f'{trial_name}: IK fitting start ({num_input_frames} frames)')
    poses_q, ik_stats = fit_q(poses, sk, ik_mapping=spec.ik_mapping)
    del poses
    per_frame_fk = ik_stats.pop('per_frame_fk_loss', None)
    per_frame_solver = ik_stats.pop('per_frame_loss', None)
    append_verbose_log(f"{trial_name}: IK fitting done mean_fk_loss={ik_stats.get('mean_fk_loss', float('nan')):.6f} mean_ik_error={ik_stats.get('mean_ik_error', float('nan')):.6f} success_ratio={ik_stats.get('success_ratio', 0.0):.4f} frames={int(ik_stats.get('total_frames', 0))}")
    if per_frame_fk is not None:
        ik_stats['per_frame_fk_loss'] = per_frame_fk
    if per_frame_solver is not None:
        ik_stats['per_frame_loss'] = per_frame_solver
    ik_stats['pose_smoothing_enabled'] = 0.0
    ik_stats['activation_method'] = 'none'
    poses_q_f32 = np.ascontiguousarray(poses_q, dtype=np.float32)
    del poses_q
    q_traj = np.ascontiguousarray(poses_q_f32.T, dtype=np.float64)
    merged_gate = gate_cfg or IkGateConfig.default()
    ik_ok, ik_reason = evaluate_ik_gate(ik_stats, q=q_traj, gate_cfg=merged_gate)
    manifest_status = 'ik_ok' if ik_ok else 'ik_failed'
    if not ik_ok:
        ik_stats['ik_gate_reason'] = ik_reason
    num_frames = int(poses_q_f32.shape[1])
    muscle_act = np.zeros((num_frames, MUSCLE_ACTIVATION_ROWS), dtype=np.float32)
    ik_stats['muscle_activation_skipped'] = 1.0
    ik_stats['muscle_activation_computed'] = 0.0
    stats, num_dofs, meta_strings = _write_b3d_from_q(poses_q_f32=poses_q_f32, output_b3d_path=output_b3d_path, trial_name=trial_name, fps=fps, mass_kg=mass_kg, height_m=height_m, sk=sk, spec=spec, muscle_act=muscle_act, sim_grf_pack=None, activation_mask_pack=np.zeros(num_frames, dtype=np.float32), ik_stats=ik_stats)
    clear_export_caches()
    return (stats, num_dofs, meta_strings, manifest_status)

def _read_poses_from_b3d(b3d_path: Path, *, trial: int=0) -> Tuple[np.ndarray, int, float, float]:
    subj = nimble.biomechanics.SubjectOnDisk(str(b3d_path))
    tlen = int(subj.getTrialLength(trial))
    kin = kinematics_pass_index(subj, trial)
    frames = subj.readFrames(trial=trial, startFrame=0, numFramesToRead=tlen, includeSensorData=False, includeProcessingPasses=True)
    poses = [np.asarray(fr.processingPasses[kin].pos, dtype=np.float32).reshape(-1) for fr in frames]
    poses_q_f32 = np.stack(poses, axis=1)
    height_m = 1.75
    mass_kg = 70.0
    try:
        height_m = float(subj.getHeightM())
        mass_kg = float(subj.getMassKg())
    except Exception:
        pass
    return (poses_q_f32, tlen, mass_kg, height_m)

def patch_b3d_moco(b3d_path: str | Path, *, trial_name: str, act_cfg: MuscleActivationConfig, ik_manifest_status: str | None=None, ik_stats: Dict[str, Any] | None=None) -> Tuple[Dict[str, float], int, Dict[str, str], str]:
    configure_opensim_logging(act_cfg.opensim_log_level)
    sk, spec = _get_skeleton(opensim_log_level=act_cfg.opensim_log_level)
    path = Path(b3d_path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing B3D cache: {path}')
    poses_q_f32, num_frames, mass_kg, height_m = _read_poses_from_b3d(path)
    subj = nimble.biomechanics.SubjectOnDisk(str(path))
    kin = kinematics_pass_index(subj, 0)
    q_traj = read_q_frames(subj, 0, 0, num_frames, kin=kin).astype(np.float64)
    stats: Dict[str, Any] = dict(ik_stats or {})
    allowed, reason = evaluate_moco_preflight_gate(ik_manifest_status=ik_manifest_status, q=q_traj)
    method = normalize_activation_method(act_cfg.activation_method)
    stats['activation_method'] = method
    sim_grf_pack: np.ndarray | None = None
    activation_mask_pack: np.ndarray | None = None
    muscle_act = np.full((num_frames, MUSCLE_ACTIVATION_ROWS), np.nan, dtype=np.float32)
    if not allowed:
        stats['moco_skipped_reason'] = reason
        stats['muscle_activation_skipped'] = 1.0
        stats['muscle_activation_computed'] = 0.0
        activation_mask_pack = np.zeros(num_frames, dtype=np.float32)
        manifest_status = 'moco_skipped'
    else:
        append_verbose_log(f'{trial_name}: {method} start ({num_frames} frames, mesh={act_cfg.mesh_interval})')
        t0 = time.perf_counter()
        try:
            act_result = compute_muscle_activation(q_traj, cfg=act_cfg)
        except Exception as exc:
            n_muscles = len(muscle_names())
            act_result = MuscleActivationResult(activations=np.full((num_frames, n_muscles), np.nan, dtype=np.float32), muscle_names=muscle_names(), metadata={'activation_method': method, 'activation_validity_mask': np.zeros(num_frames, dtype=np.float32), 'error': str(exc), 'moco_segment_success_count': 0}, forces=np.full((num_frames, SIM_GRF_COLS), np.nan, dtype=np.float32))
        stats['muscle_activation_seconds'] = float(time.perf_counter() - t0)
        stats['muscle_activation_computed'] = 1.0
        seg_count = act_result.metadata.get('moco_segment_success_count', 0)
        append_verbose_log(f"{trial_name}: {method} done seconds={stats['muscle_activation_seconds']:.1f} segment_success={seg_count}/{act_result.metadata.get('moco_segment_count', 'n/a')}")
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
            sim_grf_pack = np.asarray(act_result.forces, dtype=np.float32)
        elif act_result.metadata.get('sim_grf') is not None:
            sim_grf_pack = np.asarray(act_result.metadata['sim_grf'], dtype=np.float32)
        mask = act_result.metadata.get('activation_validity_mask')
        activation_mask_pack = np.asarray(mask, dtype=np.float32).reshape(-1) if mask is not None else np.isfinite(muscle_act).all(axis=1).astype(np.float32)
        stats['activation_valid_fraction'] = activation_valid_fraction(muscle_act, activation_mask_pack)
        stats['moco_segment_success_fraction'] = float(act_result.metadata.get('moco_segment_success_fraction', 0.0))
        manifest_status = derive_moco_manifest_status(segment_success_count=int(act_result.metadata.get('moco_segment_success_count', 0)))
        del act_result
    out_stats, num_dofs, meta_strings = _write_b3d_from_q(poses_q_f32=poses_q_f32, output_b3d_path=path, trial_name=trial_name, fps=float(act_cfg.fps), mass_kg=mass_kg, height_m=height_m, sk=sk, spec=spec, muscle_act=muscle_act, sim_grf_pack=sim_grf_pack, activation_mask_pack=activation_mask_pack, ik_stats=stats)
    clear_export_caches()
    return (out_stats, num_dofs, meta_strings, manifest_status)

def export_motion_to_b3d(hml3d_positions: np.ndarray, output_b3d_path: str | Path, *, trial_name: str, fps: float=20.0, mass_kg: float=70.0, height_m: float=1.75, muscle_activation_cfg: MuscleActivationConfig | None=None, skip_muscle_activation: bool=False, activation_method: str | None=None, gate_cfg: IkGateConfig | None=None) -> Tuple[Dict[str, float], int, Dict[str, str]]:
    act_cfg = replace(muscle_activation_cfg or MuscleActivationConfig(fps=float(fps), mass_kg=float(mass_kg)), fps=float(fps), mass_kg=float(mass_kg))
    if activation_method is not None:
        act_cfg = replace(act_cfg, activation_method=normalize_activation_method(activation_method))
    elif skip_muscle_activation:
        act_cfg = replace(act_cfg, activation_method='none')
    ik_stats, num_dofs, meta_strings, ik_status = export_ik_to_b3d(hml3d_positions, output_b3d_path, trial_name=trial_name, fps=fps, mass_kg=mass_kg, height_m=height_m, gate_cfg=gate_cfg, opensim_log_level=act_cfg.opensim_log_level)
    if normalize_activation_method(act_cfg.activation_method) == 'none':
        return (ik_stats, num_dofs, meta_strings)
    moco_stats, num_dofs, meta_strings, _ = patch_b3d_moco(output_b3d_path, trial_name=trial_name, act_cfg=act_cfg, ik_manifest_status=ik_status, ik_stats=ik_stats)
    return (moco_stats, num_dofs, meta_strings)
