"""HumanML3D joint positions -> musculoskeletal ``q`` B3D (IK at preprocess only)."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

import nimblephysics as nimble

from nimble.b3d_schema import (
    B3D_CUSTOM_VALUE_NAMES,
    MUSCLE_ACTIVATION_ROWS,
    pack_guidance_features,
    pack_muscle_activation_mask,
    pack_muscle_activations,
    pack_sindy_features,
)
from nimble.activation_gates import (
    evaluate_activation_gate,
    pelvis_ty_range_m,
    summarize_moco_metadata,
)
from common.run_logging import append_verbose_log

from nimble.muscle_activation import (
    MuscleActivationConfig,
    compute_muscle_activation,
    configure_opensim_logging,
    normalize_activation_method,
    opensim_quiet,
)
from nimble.ik import fit_q
from nimble.physics import load_model
from nimble.skeleton_registry import get_spec
from nimble.smoothing import apply_pose_smoothing_numpy
from sindy.features import features_from_q
from sindy.targets import bio_matrix, default_physics_cfg

# Post-IK temporal low-pass (same defaults as diffusion/SINDy guidance physics).
DEFAULT_SMOOTH_POSES = True
# Match OpenCap Moco reference filtering (6 Hz) at 20 fps (Nyquist 10 Hz).
DEFAULT_SMOOTH_CUTOFF_HZ = 6.0
DEFAULT_SMOOTH_BUTTERWORTH_ORDER = 2

_SKELETON_CACHE: Dict[str, Any] = {}


def _get_skeleton(*, opensim_log_level: str = "Off") -> Tuple[Any, Any]:
    sk = _SKELETON_CACHE.get("rajagopal")
    if sk is None:
        with opensim_quiet(opensim_log_level):
            sk = load_model().skeleton
        _SKELETON_CACHE["rajagopal"] = sk
    return sk, get_spec("rajagopal")


def _populate_pass_derived_values(
    kin_pass: Any,
    skeleton: Any,
    poses_q: np.ndarray,
    dt: float,
    foot_body_names: Tuple[str, str],
    *,
    root_history_len: int = 10,
    root_history_stride: int = 3,
) -> None:
    """Fill COM, velocities, accelerations, joint centers, root history, etc.

    With no force-plate data, use ``computeValuesFromForcePlates`` and an empty plate
    list (kinematics + zero GRF on the named foot bodies). Calling
    ``computeValues`` with ``zeros((3, T))`` would be parsed as one force plate
    and trigger out-of-bounds warnings.
    """
    poses = np.ascontiguousarray(poses_q, dtype=np.float64)
    kin_pass.computeValuesFromForcePlates(
        skeleton,
        float(dt),
        poses,
        list(foot_body_names),
        [],
        rootHistoryLen=int(root_history_len),
        rootHistoryStride=int(root_history_stride),
    )


def export_motion_to_b3d(
    hml3d_positions: np.ndarray,
    output_b3d_path: str | Path,
    *,
    trial_name: str,
    fps: float = 20.0,
    mass_kg: float = 70.0,
    height_m: float = 1.75,
    muscle_activation_cfg: MuscleActivationConfig | None = None,
    skip_muscle_activation: bool = False,
    activation_method: str | None = None,
) -> Tuple[Dict[str, float], int, Dict[str, str]]:
    """Fit Rajagopal ``q`` and write a single-trial kinematics B3D file.

    Always fills derived kinematics (COM, velocities, joint centers, zero GRF
    on the skeleton's foot bodies).

    Returns:
        (ik_stats, num_dofs, meta_strings)
    """
    poses = np.asarray(hml3d_positions, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (22, 3):
        raise ValueError(f"Expected hml3d_positions [T,22,3], got {poses.shape}")

    act_cfg_early = muscle_activation_cfg or MuscleActivationConfig(
        fps=float(fps), mass_kg=float(mass_kg)
    )
    configure_opensim_logging(act_cfg_early.opensim_log_level)

    sk, spec = _get_skeleton(opensim_log_level=act_cfg_early.opensim_log_level)
    foot_body_names = tuple(spec.foot_body_names)
    num_input_frames = int(poses.shape[0])
    append_verbose_log(f"{trial_name}: IK fitting start ({num_input_frames} frames)")
    poses_q, ik_stats = fit_q(poses, sk, ik_mapping=spec.ik_mapping)
    append_verbose_log(
        f"{trial_name}: IK fitting done "
        f"mean_fk_loss={ik_stats.get('mean_fk_loss', float('nan')):.6f} "
        f"mean_ik_error={ik_stats.get('mean_ik_error', float('nan')):.6f} "
        f"success_ratio={ik_stats.get('success_ratio', 0.0):.4f} "
        f"frames={int(ik_stats.get('total_frames', 0))}"
    )

    min_success = 2
    if int(ik_stats.get("success_count", 0)) < min_success:
        raise RuntimeError(
            f"IK failed for {trial_name}: success {ik_stats.get('success_count', 0)}/"
            f"{ik_stats.get('total_frames', 0)}"
        )

    q_time_dof = np.ascontiguousarray(poses_q.T, dtype=np.float32)
    q_sm, smooth_meta = apply_pose_smoothing_numpy(
        q_time_dof,
        fps=float(fps),
        smooth_poses=DEFAULT_SMOOTH_POSES,
        smooth_cutoff_hz=DEFAULT_SMOOTH_CUTOFF_HZ,
        smooth_butterworth_order=DEFAULT_SMOOTH_BUTTERWORTH_ORDER,
    )
    poses_q = np.ascontiguousarray(q_sm.T, dtype=np.float64)
    ik_stats["pose_smoothing_enabled"] = float(bool(smooth_meta.get("enabled", False)))
    eff_hz = smooth_meta.get("cutoff_hz_effective")
    if eff_hz is not None:
        ik_stats["pose_smooth_cutoff_hz"] = float(eff_hz)

    poses_q_f32 = np.ascontiguousarray(poses_q, dtype=np.float32)
    dt = 1.0 / max(float(fps), 1e-6)
    num_dofs = int(sk.getNumDofs())

    b3d_subject = nimble.biomechanics.SubjectOnDiskHeader()
    b3d_subject_pass = b3d_subject.addProcessingPass()
    b3d_subject_pass.setProcessingPassType(nimble.biomechanics.ProcessingPassType.KINEMATICS)
    b3d_subject.setHeightM(float(height_m))
    b3d_subject.setMassKg(float(mass_kg))
    b3d_subject.setNumDofs(num_dofs)
    b3d_subject.setNumJoints(int(sk.getNumJoints()))
    b3d_subject.setGroundForceBodies(list(foot_body_names))
    b3d_subject.setNotes(f"Converted from HumanML3D motion: {trial_name}")

    b3d_trial = b3d_subject.addTrial()
    b3d_trial.setName(str(trial_name))
    b3d_trial.setTrialLength(int(poses_q_f32.shape[1]))
    b3d_trial.setTimestep(float(dt))
    b3d_trial.setForcePlates([])

    kin_pass = b3d_trial.addPass()
    kin_pass.setType(nimble.biomechanics.ProcessingPassType.KINEMATICS)
    kin_pass.setPoses(poses_q_f32)
    # nimblephysics' writeB3D uses len(MarkerObservations) as the per-trial frame
    # count; passing [] writes a 0-byte file even when poses are set. Provide one
    # empty per-frame dict so the writer emits all T frames.
    num_frames = int(poses_q_f32.shape[1])
    b3d_trial.setMarkerObservations([{} for _ in range(num_frames)])

    _populate_pass_derived_values(
        kin_pass,
        sk,
        poses_q_f32,
        dt,
        foot_body_names,
        root_history_len=10,
        root_history_stride=3,
    )

    q_traj = np.ascontiguousarray(poses_q_f32.T, dtype=np.float64)
    bio_cfg = default_physics_cfg(fps=float(fps), max_frames=int(q_traj.shape[0]))
    bio = bio_matrix(q_traj, fps=float(fps), guidance_cfg=bio_cfg)
    u, c, _, _ = features_from_q(q_traj, sk, fps=float(fps))

    num_frames = int(q_traj.shape[0])
    act_cfg = muscle_activation_cfg or MuscleActivationConfig(
        fps=float(fps), mass_kg=float(mass_kg)
    )
    act_cfg = replace(act_cfg, fps=float(fps), mass_kg=float(mass_kg))
    if activation_method is not None:
        act_cfg = replace(
            act_cfg,
            activation_method=normalize_activation_method(activation_method),
        )
    elif skip_muscle_activation:
        act_cfg = replace(act_cfg, activation_method="none")

    method = str(act_cfg.activation_method)
    ik_stats["activation_method"] = method

    if method == "none":
        ik_stats["muscle_activation_skipped"] = 1.0
        ik_stats["muscle_activation_computed"] = 0.0
        muscle_act = np.zeros((num_frames, MUSCLE_ACTIVATION_ROWS), dtype=np.float64)
        muscle_mask = np.zeros(num_frames, dtype=np.float64)
    else:
        ik_stats["pelvis_ty_range_m"] = pelvis_ty_range_m(q_traj)
        gate_ok, gate_reason = evaluate_activation_gate(
            ik_stats,
            num_frames=num_frames,
            cfg=act_cfg,
        )
        ik_stats["activation_gate_passed"] = float(gate_ok)
        if gate_reason:
            ik_stats["activation_gate_reason"] = gate_reason
        if not gate_ok:
            raise RuntimeError(
                f"Muscle activation skipped ({method}): IK/clip gate failed ({gate_reason})"
            )

        append_verbose_log(
            f"{trial_name}: {method} start "
            f"({num_frames} frames, mesh={act_cfg.mesh_interval})"
        )
        t0 = time.perf_counter()
        act_result = compute_muscle_activation(q_traj, cfg=act_cfg)
        ik_stats["muscle_activation_seconds"] = float(time.perf_counter() - t0)
        obj = act_result.metadata.get("moco_objective")
        obj_s = f"{float(obj):.6f}" if obj is not None and np.isfinite(float(obj)) else "n/a"
        append_verbose_log(
            f"{trial_name}: {method} done "
            f"seconds={ik_stats['muscle_activation_seconds']:.1f} "
            f"label_valid_fraction={act_result.metadata.get('label_valid_fraction', 0.0):.4f} "
            f"objective={obj_s} "
            f"solver_success={act_result.metadata.get('moco_solver_success')}"
        )
        solve_details = act_result.metadata.get("moco_solve_details")
        if isinstance(solve_details, dict):
            iters = solve_details.get("solver_iterations")
            status = solve_details.get("solver_status")
            mesh = solve_details.get("mesh_interval")
            if iters is not None or status or mesh is not None:
                append_verbose_log(
                    f"{trial_name}: moco_solve "
                    f"iterations={iters} status={status} mesh_interval={mesh}"
                )
        ik_stats["muscle_activation_computed"] = 1.0
        label_frac = float(act_result.metadata.get("label_valid_fraction", 1.0))
        ik_stats["muscle_activation_label_valid_fraction"] = label_frac
        ik_stats["muscle_activation_success_fraction"] = label_frac
        ik_stats["muscle_activation_repaired_frames"] = float(
            act_result.metadata.get("repaired_frame_count", 0)
        )
        moco_obj = act_result.metadata.get("moco_objective")
        if moco_obj is not None and np.isfinite(float(moco_obj)):
            ik_stats["muscle_activation_objective"] = float(moco_obj)
        for key, val in summarize_moco_metadata(act_result.metadata).items():
            if isinstance(val, (int, float)):
                ik_stats[f"moco_{key}"] = float(val)
        muscle_act = act_result.activations
        muscle_mask = act_result.label_valid

    b3d_subject.setCustomValueNames(list(B3D_CUSTOM_VALUE_NAMES))
    b3d_trial.setCustomValues(
        [
            pack_guidance_features(bio),
            pack_sindy_features(u, c),
            pack_muscle_activations(muscle_act),
            pack_muscle_activation_mask(muscle_mask),
        ]
    )
    ik_stats["guidance_features_computed"] = 1.0
    ik_stats["sindy_features_computed"] = 1.0

    out_path = Path(output_b3d_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nimble.biomechanics.SubjectOnDisk.writeB3D(str(out_path), b3d_subject)

    stats: Dict[str, float] = {}
    meta_strings: Dict[str, str] = {}
    for k, v in ik_stats.items():
        if isinstance(v, str):
            meta_strings[k] = v
        elif isinstance(v, (int, float, np.integer, np.floating)):
            stats[k] = float(v)
        elif isinstance(v, bool):
            stats[k] = float(v)
    return stats, num_dofs, meta_strings
