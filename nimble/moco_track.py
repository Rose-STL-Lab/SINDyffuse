"""OpenSim MocoTrack muscle activations from Nimble Rajagopal ``q``.

Uses ``MocoTrack`` with soft coordinate tracking and foot–ground contact
(``SmoothSphereHalfSpaceForce`` on foot bodies). Each clip is solved in one
pass across all frames.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import opensim as osim

from common.working_directory import working_directory

from nimble.muscle_activation import (
    MuscleActivationConfig,
    MuscleActivationResult,
    apply_activation_postprocess,
    _storage_to_array,
    muscle_names,
    opensim_quiet,
    rajagopal_model_path,
)
from nimble.rajagopal_model import (
    build_activation_model_processor,
    muscle_names_from_processor,
    unlock_rajagopal_coordinates,
)
from nimble.rajagopal_coord_map import (
    build_moco_states_table_processor,
    build_rajagopal_coord_mapping,
    write_coordinates_mot,
)

_MOCO_TRACK_MODEL_SINGLE = "rajagopal_moco_track.osim"
_MOCO_TRACK_MODEL_MULTI = "rajagopal_moco_track_multi.osim"

_MOCO_SIDE_EFFECT_GLOBS = (
    "delete_this_to_stop_optimization__*.txt",
    "*_tracked_states.sto",
)


def _cleanup_moco_side_effects(directory: Path) -> None:
    for pattern in _MOCO_SIDE_EFFECT_GLOBS:
        for path in directory.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass


def _sweep_repo_root_moco_artifacts() -> None:
    cwd = Path(os.getcwd())
    _cleanup_moco_side_effects(cwd)

_MOCO_MOD_OPS: Tuple[str, ...] = (
    "ModOpReplaceJointsWithWelds(mtp)",
    "ModOpIgnoreTendonCompliance",
    "ModOpReplaceMusclesWithDeGrooteFregly2016",
    "ModOpIgnorePassiveFiberForcesDGF",
    "ModOpScaleActiveFiberForceCurveWidthDGF",
    "ModOpAddResiduals",
    "ModOpAddReserves",
)

_MOCO_TOE_JOINTS: Tuple[str, ...] = ("mtp_r", "mtp_l")


@dataclass(frozen=True)
class FootContactSphereSpec:
    """One smooth sphere–half-space contact on a foot body."""

    body_name: str
    radius_m: float
    offset_x_m: float = 0.0
    offset_y_m: float = -0.02
    offset_z_m: float = 0.0


@dataclass
class MocoTrackConfig:
    """Foot–ground contact parameters for Rajagopal MocoTrack."""

    contact_spheres: Tuple[FootContactSphereSpec, ...] = ()
    stiffness_N_per_m: float = 1e5
    dissipation_Ns_per_m: float = 200.0
    multi_contact: bool = True


def _default_foot_contact_spheres(
    cfg: MuscleActivationConfig,
) -> Tuple[FootContactSphereSpec, ...]:
    r = float(cfg.moco_contact_sphere_radius_m)
    y = float(cfg.moco_contact_sphere_offset_y_m)
    calcn = (
        FootContactSphereSpec("calcn_r", r, offset_y_m=y),
        FootContactSphereSpec("calcn_l", r, offset_y_m=y),
    )
    if not bool(cfg.moco_multi_contact):
        return calcn
    toe_r = float(cfg.moco_contact_toe_radius_m)
    return calcn + (
        FootContactSphereSpec("toes_r", toe_r, offset_y_m=y * 0.75, offset_z_m=0.06),
        FootContactSphereSpec("toes_l", toe_r, offset_y_m=y * 0.75, offset_z_m=0.06),
    )


def _track_config(cfg: MuscleActivationConfig) -> MocoTrackConfig:
    multi = bool(cfg.moco_multi_contact)
    return MocoTrackConfig(
        contact_spheres=_default_foot_contact_spheres(cfg),
        stiffness_N_per_m=float(cfg.moco_contact_stiffness),
        dissipation_Ns_per_m=float(cfg.moco_contact_dissipation),
        multi_contact=multi,
    )


def _moco_track_model_filename(track_cfg: MocoTrackConfig) -> str:
    return _MOCO_TRACK_MODEL_MULTI if track_cfg.multi_contact else _MOCO_TRACK_MODEL_SINGLE


def _base_mesh_interval(cfg: MuscleActivationConfig) -> float:
    if cfg.mesh_interval is not None and float(cfg.mesh_interval) > 0:
        return float(cfg.mesh_interval)
    dt = 1.0 / max(float(cfg.fps), 1e-8)
    return min(dt, 0.2)


def _max_joint_speed_deg_s(q: np.ndarray, fps: float) -> float:
    """Peak absolute joint speed across DOFs (deg/s), for adaptive mesh selection."""
    arr = np.asarray(q, dtype=np.float64)
    if arr.shape[0] < 2:
        return 0.0
    dt = 1.0 / max(float(fps), 1e-8)
    speeds_rad_s = np.abs(np.diff(arr, axis=0)).max(axis=1) / dt
    return float(np.degrees(np.max(speeds_rad_s)))


def _effective_mesh_interval(
    cfg: MuscleActivationConfig,
    q: np.ndarray | None = None,
) -> float:
    """Mesh interval for the trial; optionally tighten for high-speed motion."""
    base = _base_mesh_interval(cfg)
    if not bool(cfg.moco_adaptive_mesh) or q is None:
        return base
    speed = _max_joint_speed_deg_s(q, float(cfg.fps))
    if speed >= float(cfg.moco_adaptive_mesh_speed_deg_s):
        return min(base, float(cfg.moco_adaptive_mesh_interval))
    return base


_unlock_locked_coordinates = unlock_rajagopal_coordinates


def _add_foot_contact(model: osim.Model, track_cfg: MocoTrackConfig) -> None:
    """Add a shared ground half-space and smooth sphere contacts per foot."""
    ground = model.getGround()
    half_space = osim.ContactHalfSpace(
        osim.Vec3(0, 0, 0),
        osim.Vec3(0, 0, -np.pi / 2.0),
        ground,
        "floor",
    )
    model.addContactGeometry(half_space)

    for spec in track_cfg.contact_spheres:
        body = model.getBodySet().get(spec.body_name)
        geom_name = f"contact_{spec.body_name}"
        sphere = osim.ContactSphere(
            float(spec.radius_m),
            osim.Vec3(
                float(spec.offset_x_m),
                float(spec.offset_y_m),
                float(spec.offset_z_m),
            ),
            body,
            geom_name,
        )
        model.addContactGeometry(sphere)
        force = osim.SmoothSphereHalfSpaceForce(
            f"contact_force_{spec.body_name}",
            sphere,
            half_space,
        )
        force.set_stiffness(float(track_cfg.stiffness_N_per_m))
        force.set_dissipation(float(track_cfg.dissipation_Ns_per_m))
        model.addForce(force)

    model.finalizeConnections()


def prepare_rajagopal_moco_track_model(
    work_dir: Path,
    *,
    model_path: Path | None = None,
    track_cfg: MocoTrackConfig | None = None,
) -> Path:
    """Write a MocoTrack-ready Rajagopal copy (unlocked coords + foot contact)."""
    if track_cfg is None:
        track_cfg = MocoTrackConfig(
            contact_spheres=_default_foot_contact_spheres(
                MuscleActivationConfig()
            ),
            multi_contact=True,
        )
    src = model_path or rajagopal_model_path()
    out = work_dir / _moco_track_model_filename(track_cfg)
    if out.is_file():
        return out

    model = osim.Model(str(src))
    _unlock_locked_coordinates(model)
    _add_foot_contact(model, track_cfg)
    model.initSystem()
    model.printToXML(str(out))
    return out


def _trial_time_bounds(t_len: int, fps: float) -> Tuple[float, float]:
    """MocoTrack time range compatible with a ``t_len``-row coordinate ``.mot``."""
    dt = 1.0 / max(float(fps), 1e-8)
    t0 = 0.0
    # Moco requires final_time strictly inside the reference table span.
    t1 = max(t0, (t_len - 1) * dt - 1e-6)
    return t0, t1


def _build_model_processor_track(
    moco_model_path: Path,
    cfg: MuscleActivationConfig,
) -> osim.ModelProcessor:
    return build_activation_model_processor(
        moco_model_path, cfg, weld_toe_joints=True
    )


_muscle_names_from_processor = muscle_names_from_processor


def _activation_column_for_muscle(label: str, muscle: str) -> bool:
    """Match OpenSim state paths like ``/forceset/gastroc_r/activation``."""
    needle = f"/{muscle}/activation"
    return label.endswith(needle) or needle in label


def _parse_moco_activation_storage(
    storage_path: Path,
    muscle_name_list: Sequence[str],
    frame_times: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """Resample Moco state activations onto uniform ``frame_times`` (seconds)."""
    storage = osim.Storage(str(storage_path))
    data, labels = _storage_to_array(storage)
    if not labels or data.size == 0:
        raise RuntimeError(f"Empty Moco solution storage: {storage_path}")

    label_to_col = {lab: i for i, lab in enumerate(labels)}
    if "time" in label_to_col:
        sol_times = data[:, label_to_col["time"]]
    else:
        sol_times = data[:, 0]

    n_frames = int(frame_times.shape[0])
    n_muscles = len(muscle_name_list)
    activations = np.full((n_frames, n_muscles), np.nan, dtype=np.float64)

    for mi, name in enumerate(muscle_name_list):
        col_idx: int | None = None
        for lab, idx in label_to_col.items():
            if _activation_column_for_muscle(lab, name):
                col_idx = idx
                break
        if col_idx is None:
            continue
        series = data[:, col_idx]
        finite = np.isfinite(sol_times) & np.isfinite(series)
        if finite.sum() < 2:
            continue
        activations[:, mi] = np.interp(
            frame_times,
            sol_times[finite],
            series[finite],
            left=np.nan,
            right=np.nan,
        )

    row_finite = np.isfinite(activations).all(axis=1)
    parsed_ok = bool(row_finite.any()) and np.isfinite(activations).any()
    return activations.astype(np.float32), parsed_ok


def _is_reserve_control_name(name: str) -> bool:
    lower = name.lower()
    return "reserve" in lower and "residual" not in lower


def _analyze_moco_reserve_controls(
    moco_sol: osim.MocoSolution,
    frame_times: np.ndarray,
    *,
    max_fraction: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Resample max reserve |control| onto ``frame_times``; OpenSim uses [-1, 1]."""
    names = list(moco_sol.getControlNames())
    mat = moco_sol.getControlMat()
    time_mat = moco_sol.getTimeMat()
    sol_times = np.array([float(time_mat.get(i, 0)) for i in range(time_mat.nrow())])
    n_frames = int(frame_times.shape[0])
    reserve_peaks: Dict[str, float] = {}
    combined = np.zeros(sol_times.shape[0], dtype=np.float64)
    for idx, name in enumerate(names):
        if not _is_reserve_control_name(name):
            continue
        col = np.array(
            [abs(float(mat.get(i, idx))) for i in range(mat.nrow())],
            dtype=np.float64,
        )
        combined = np.maximum(combined, col)
        short = name.rsplit("/", 1)[-1]
        reserve_peaks[short] = float(np.max(col)) if col.size else 0.0

    if combined.size == 0:
        by_time = np.zeros(n_frames, dtype=np.float64)
        meta = {
            "max_reserve_fraction": 0.0,
            "reserve_qc_pass": True,
            "reserve_control_peaks": {},
        }
        return by_time, meta

    finite = np.isfinite(sol_times) & np.isfinite(combined)
    if finite.sum() < 2:
        by_time = np.full(n_frames, float(np.nanmax(combined)), dtype=np.float64)
    else:
        by_time = np.interp(
            frame_times,
            sol_times[finite],
            combined[finite],
            left=combined[finite][0],
            right=combined[finite][-1],
        )

    max_frac = float(np.max(by_time)) if by_time.size else 0.0
    meta = {
        "max_reserve_fraction": max_frac,
        "reserve_qc_pass": max_frac <= float(max_fraction),
        "reserve_control_peaks": reserve_peaks,
    }
    return by_time, meta


def _configure_moco_state_tracking_weights(
    study: osim.MocoStudy,
    *,
    cfg: MuscleActivationConfig,
) -> None:
    """Uniform coordinate tracking for full-body motion; free ``pelvis_ty`` for contact."""
    problem = study.updProblem()
    try:
        goal = osim.MocoStateTrackingGoal.safeDownCast(problem.updGoal("state_tracking"))
    except Exception:
        return

    pos_w = float(cfg.moco_states_tracking_weight)
    speed_w = float(cfg.moco_states_speed_tracking_weight)

    try:
        model = problem.getModelBase().process()
        model.initSystem()
        state_names = model.getStateVariableNames()
    except Exception:
        return

    for i in range(state_names.size()):
        path = str(state_names.get(i))
        if not path.startswith("/jointset/"):
            continue
        if path.endswith("/pelvis_ty/value"):
            goal.setWeightForState(path, 0.0)
        elif path.endswith("/speed"):
            goal.setWeightForState(path, speed_w)
        else:
            goal.setWeightForState(path, pos_w)


def _configure_moco_track_solver(
    study: osim.MocoStudy,
    *,
    cfg: MuscleActivationConfig,
) -> None:
    problem = study.updProblem()
    effort = osim.MocoControlGoal.safeDownCast(problem.updGoal("control_effort"))
    effort.setWeight(float(cfg.moco_control_effort_weight))
    effort.setWeightForControlPattern(".*reserve.*", float(cfg.moco_reserve_control_weight))
    effort.setWeightForControlPattern(".*pelvis.*", 10.0)

    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
    solver.set_optim_convergence_tolerance(float(cfg.moco_convergence_tolerance))
    solver.set_optim_constraint_tolerance(float(cfg.moco_convergence_tolerance))
    if int(cfg.moco_max_iterations) > 0:
        solver.set_optim_max_iterations(int(cfg.moco_max_iterations))
    if bool(cfg.moco_minimize_implicit_aux_derivatives):
        solver.set_minimize_implicit_auxiliary_derivatives(True)
        solver.set_implicit_auxiliary_derivatives_weight(
            float(cfg.moco_implicit_aux_derivatives_weight)
        )
    _configure_moco_state_tracking_weights(study, cfg=cfg)
    solver.resetProblem(problem)


def _solve_moco_track(
    q: np.ndarray,
    *,
    cfg: MuscleActivationConfig,
    solve_dir: Path,
    moco_model_path: Path,
    mapping: Any,
    muscle_name_list: Sequence[str],
    mesh_interval: float,
) -> Tuple[np.ndarray, bool, Dict[str, Any], Path, np.ndarray]:
    """Run one MocoTrack solve across all frames."""
    t_len = int(q.shape[0])
    dt = 1.0 / max(float(cfg.fps), 1e-8)
    frame_times = np.arange(t_len, dtype=np.float64) * dt
    t0, t1 = _trial_time_bounds(t_len, float(cfg.fps))

    mot_path = solve_dir / "coordinates.mot"
    write_coordinates_mot(q, mot_path, fps=float(cfg.fps), mapping=mapping)

    mp = _build_model_processor_track(moco_model_path, cfg)
    out_sto = solve_dir / "moco_track_solution.sto"

    track = osim.MocoTrack()
    track.setName("moco_track")
    track.setModel(mp)
    track.setStatesReference(
        build_moco_states_table_processor(
            mot_path,
            lowpass_hz=float(cfg.moco_reference_lowpass_hz),
        )
    )
    track.set_allow_unused_references(True)
    track.set_track_reference_position_derivatives(True)
    track.set_states_global_tracking_weight(1.0)
    track.set_control_effort_weight(float(cfg.moco_control_effort_weight))
    if bool(cfg.moco_apply_tracked_states_to_guess):
        track.set_apply_tracked_states_to_guess(True)
    track.set_initial_time(t0)
    track.set_final_time(t1)
    track.set_mesh_interval(float(mesh_interval))

    solve_meta: Dict[str, Any] = {
        "num_frames": t_len,
        "t0": t0,
        "t1": t1,
        "mesh_interval": float(mesh_interval),
        "moco_reference_lowpass_hz": float(cfg.moco_reference_lowpass_hz),
        "moco_states_speed_tracking_weight": float(cfg.moco_states_speed_tracking_weight),
        "moco_aux_coord_tracking_weight": float(cfg.moco_aux_coord_tracking_weight),
        "moco_apply_tracked_states_to_guess": bool(cfg.moco_apply_tracked_states_to_guess),
        "max_joint_speed_deg_s": _max_joint_speed_deg_s(q, float(cfg.fps)),
    }

    reserve_by_time = np.zeros(t_len, dtype=np.float64)
    try:
        with working_directory(solve_dir.resolve()):
            study = track.initialize()
            _configure_moco_track_solver(study, cfg=cfg)
            moco_sol = study.solve()
            solve_meta["solver_success"] = bool(moco_sol.success())
            try:
                solve_meta["solver_status"] = str(moco_sol.getStatus())
                solve_meta["solver_iterations"] = int(moco_sol.getNumIterations())
            except Exception:
                pass
            if not moco_sol.success():
                moco_sol.unseal()
            moco_sol.write(str(out_sto.resolve()))
        _cleanup_moco_side_effects(solve_dir)
        activations, parsed_ok = _parse_moco_activation_storage(
            out_sto, muscle_name_list, frame_times
        )
        reserve_by_time, reserve_meta = _analyze_moco_reserve_controls(
            moco_sol,
            frame_times,
            max_fraction=float(cfg.moco_max_reserve_fraction),
        )
        solve_meta.update(reserve_meta)
        solve_ok = bool(moco_sol.success()) and parsed_ok
        solve_meta["success"] = solve_ok
        try:
            solve_meta["objective"] = float(moco_sol.getObjective())
        except Exception:
            pass
    except Exception as exc:
        activations = np.full((t_len, len(muscle_name_list)), np.nan, dtype=np.float32)
        solve_ok = False
        solve_meta["success"] = False
        solve_meta["error"] = str(exc)

    return activations, solve_ok, solve_meta, out_sto, reserve_by_time


def run_moco_track(
    q: np.ndarray,
    *,
    cfg: MuscleActivationConfig,
    work_dir: Path,
) -> MuscleActivationResult:
    """Run MocoTrack across all frames and return activations ``[T, M]``."""
    _sweep_repo_root_moco_artifacts()
    arr = np.asarray(q, dtype=np.float64)
    t_len = int(arr.shape[0])
    if t_len < 2:
        raise ValueError(f"Need at least 2 frames for MocoTrack, got {t_len}")

    model_path = rajagopal_model_path()
    track_cfg = _track_config(cfg)
    solve_dir = work_dir / "moco_track"
    solve_dir.mkdir(parents=True, exist_ok=True)
    mesh_interval = _effective_mesh_interval(cfg, arr)

    with opensim_quiet(cfg.opensim_log_level):
        mapping = build_rajagopal_coord_mapping(model_path=model_path)
        names_ref = muscle_names()
        moco_model_path = prepare_rajagopal_moco_track_model(
            work_dir,
            model_path=model_path,
            track_cfg=track_cfg,
        )
        activations, solve_ok, solve_meta, _, _reserve_by_time = _solve_moco_track(
            arr,
            cfg=cfg,
            solve_dir=solve_dir,
            moco_model_path=moco_model_path,
            mapping=mapping,
            muscle_name_list=names_ref,
            mesh_interval=mesh_interval,
        )

    if activations.shape[0] != t_len:
        raise RuntimeError(
            f"Moco activation length {activations.shape[0]} != expected {t_len}"
        )
    if activations.shape[1] != len(names_ref):
        raise RuntimeError(
            f"Moco muscle count {activations.shape[1]} != expected {len(names_ref)}"
        )

    repair_meta: Dict[str, Any] = {}
    if cfg.interpolate_activations or float(cfg.activation_smooth_hz) > 0.0:
        activations, repair_meta = apply_activation_postprocess(activations, cfg)
    repaired_frame_count = int(repair_meta.get("repaired_frame_count", 0))

    obj = solve_meta.get("objective")
    moco_objective = float(obj) if obj is not None and np.isfinite(float(obj)) else float("nan")

    meta: Dict[str, Any] = {
        "activation_method": "moco_track",
        "opensim_version": str(osim.GetVersion()),
        "moco_version": str(osim.GetMocoVersionAndDate())
        if hasattr(osim, "GetMocoVersionAndDate")
        else "",
        "model_path": str(model_path),
        "moco_model_path": str(moco_model_path),
        "num_frames": t_len,
        "num_muscles": len(names_ref),
        "fps": float(cfg.fps),
        "mesh_interval": mesh_interval,
        "moco_solver_success": bool(solve_meta.get("solver_success", solve_ok)),
        "moco_objective": moco_objective,
        "repaired_frame_count": repaired_frame_count,
        "moco_contact": True,
        "moco_multi_contact": bool(track_cfg.multi_contact),
        "moco_contact_bodies": [s.body_name for s in track_cfg.contact_spheres],
        "moco_adaptive_mesh": bool(cfg.moco_adaptive_mesh),
        "moco_states_tracking_weight": float(cfg.moco_states_tracking_weight),
        "moco_states_speed_tracking_weight": float(cfg.moco_states_speed_tracking_weight),
        "moco_aux_coord_tracking_weight": float(cfg.moco_aux_coord_tracking_weight),
        "moco_reference_lowpass_hz": float(cfg.moco_reference_lowpass_hz),
        "moco_apply_tracked_states_to_guess": bool(cfg.moco_apply_tracked_states_to_guess),
        "moco_minimize_implicit_aux_derivatives": bool(
            cfg.moco_minimize_implicit_aux_derivatives
        ),
        "moco_control_effort_weight": float(cfg.moco_control_effort_weight),
        "moco_reserve_control_weight": float(cfg.moco_reserve_control_weight),
        "moco_max_reserve_fraction": float(cfg.moco_max_reserve_fraction),
        "moco_fail_on_high_reserve": bool(cfg.moco_fail_on_high_reserve),
        "moco_residual_force": float(cfg.moco_residual_force),
        "moco_reserve_optimal_force": float(cfg.moco_reserve_optimal_force),
        "moco_reserve_scale": float(cfg.moco_reserve_scale),
        "moco_mod_ops": list(_MOCO_MOD_OPS),
        "moco_solve_details": solve_meta,
    }
    meta.update(repair_meta)
    if solve_meta.get("solver_status"):
        meta["moco_solver_status"] = str(solve_meta["solver_status"])
    if solve_meta.get("solver_iterations") is not None:
        meta["moco_solver_iterations"] = int(solve_meta["solver_iterations"])

    return MuscleActivationResult(
        activations=activations,
        muscle_names=tuple(names_ref),
        metadata=meta,
    )
