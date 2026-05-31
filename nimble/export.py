"""HumanML3D joint positions -> musculoskeletal ``q`` B3D (IK at preprocess only)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

import nimblephysics as nimble

from nimble.b3d_schema import (
    B3D_CUSTOM_VALUE_NAMES,
    pack_guidance_features,
    pack_muscle_activations,
    pack_sindy_features,
)
from surrogate.opensim_activation import MuscleActivationConfig, compute_muscle_activation
from nimble.ik import fit_q
from nimble.physics import load_model
from nimble.skeleton_registry import get_spec
from nimble.smoothing import apply_pose_smoothing_numpy
from sindy.features import features_from_q
from sindy.targets import bio_matrix, default_physics_cfg

# Post-IK temporal low-pass (same defaults as diffusion/SINDy guidance physics).
DEFAULT_SMOOTH_POSES = True
# Slightly below guidance default (6 Hz) to reduce per-frame IK jitter in offline B3D export.
DEFAULT_SMOOTH_CUTOFF_HZ = 4.0
DEFAULT_SMOOTH_BUTTERWORTH_ORDER = 2

_SKELETON_CACHE: Dict[str, Any] = {}


def _get_skeleton() -> Tuple[Any, Any]:
    sk = _SKELETON_CACHE.get("rajagopal")
    if sk is None:
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
) -> Tuple[Dict[str, float], int]:
    """Fit Rajagopal ``q`` and write a single-trial kinematics B3D file.

    Always fills derived kinematics (COM, velocities, joint centers, zero GRF
    on the skeleton's foot bodies).

    Returns:
        (ik_stats, num_dofs)
    """
    poses = np.asarray(hml3d_positions, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (22, 3):
        raise ValueError(f"Expected hml3d_positions [T,22,3], got {poses.shape}")

    sk, spec = _get_skeleton()
    foot_body_names = tuple(spec.foot_body_names)
    poses_q, ik_stats = fit_q(poses, sk, ik_mapping=spec.ik_mapping)

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

    act_cfg = MuscleActivationConfig(fps=float(fps), mass_kg=float(mass_kg))
    t0 = time.perf_counter()
    act_result = compute_muscle_activation(q_traj, cfg=act_cfg)
    ik_stats["muscle_activation_seconds"] = float(time.perf_counter() - t0)
    ik_stats["muscle_activation_computed"] = 1.0
    ik_stats["muscle_activation_success_fraction"] = float(
        act_result.metadata.get("success_fraction", 1.0)
    )

    b3d_subject.setCustomValueNames(list(B3D_CUSTOM_VALUE_NAMES))
    b3d_trial.setCustomValues(
        [
            pack_guidance_features(bio),
            pack_sindy_features(u, c),
            pack_muscle_activations(act_result.activations),
        ]
    )
    ik_stats["guidance_features_computed"] = 1.0
    ik_stats["sindy_features_computed"] = 1.0

    out_path = Path(output_b3d_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nimble.biomechanics.SubjectOnDisk.writeB3D(str(out_path), b3d_subject)

    stats: Dict[str, float] = {}
    for k, v in ik_stats.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            stats[k] = float(v)
        elif isinstance(v, bool):
            stats[k] = float(v)
    return stats, num_dofs
