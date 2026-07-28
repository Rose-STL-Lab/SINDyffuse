"""OpenSim MocoTrack muscle activations from Nimble Rajagopal ``q``.

Uses ``MocoTrack`` with soft coordinate tracking and foot–ground contact
(``SmoothSphereHalfSpaceForce`` on foot bodies). Clips are solved in MinT-style
1.4 s segmented windows via ``nimble.moco_segment``.
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
    _storage_to_array,
    activation_column_for_muscle,
    muscle_names,
    opensim_quiet,
    rajagopal_model_path,
)
from nimble.rajagopal_model import (
    build_activation_model_processor,
    muscle_names_from_processor,
    unlock_rajagopal_coordinates,
)
from nimble.moco_segment import SIM_GRF_COLS
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
            if activation_column_for_muscle(lab, name):
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


def _side_from_contact_force_name(name: str) -> str:
    lower = str(name).lower()
    if "_l" in lower or lower.endswith("left"):
        return "left"
    return "right"


def _extract_sim_grf_from_moco_solution(
    moco_sol: osim.MocoSolution,
    moco_model_path: Path,
    frame_times: np.ndarray,
    track_cfg: MocoTrackConfig,
) -> np.ndarray:
    """Resample simulated GRF from Moco states onto ``frame_times`` (``[T, 18]``)."""
    n_frames = int(frame_times.shape[0])
    grf = np.full((n_frames, SIM_GRF_COLS), np.nan, dtype=np.float64)
    try:
        model = osim.Model(str(moco_model_path))
        model.initSystem()
        state = model.getWorkingState()
        states_table = moco_sol.exportToStatesTrajectoryTable()
        n_rows = int(states_table.getNumRows())
        if n_rows < 1:
            return grf.astype(np.float32)

        contact_forces: List[Tuple[str, Any]] = []
        for i in range(model.getForceSet().getSize()):
            force = model.getForceSet().get(i)
            if force.getConcreteClassName() != "SmoothSphereHalfSpaceForce":
                continue
            ssf = osim.SmoothSphereHalfSpaceForce.safeDownCast(force)
            if ssf is not None:
                contact_forces.append((force.getName(), ssf))

        sol_times = np.zeros(n_rows, dtype=np.float64)
        left_forces = np.zeros((n_rows, 3), dtype=np.float64)
        right_forces = np.zeros((n_rows, 3), dtype=np.float64)
        indep = states_table.getIndependentColumn()
        for row in range(n_rows):
            states_table.getRowAtIndex(row, state.getY())
            sol_times[row] = float(indep.get(row))
            model.realizeAcceleration(state)
            for name, ssf in contact_forces:
                vec = osim.Vector()
                ssf.computeValues(state, vec)
                n_comp = min(3, int(vec.size()))
                f = np.array([float(vec.get(j)) for j in range(n_comp)], dtype=np.float64)
                if f.size < 3:
                    f = np.pad(f, (0, 3 - f.size))
                if _side_from_contact_force_name(name) == "left":
                    left_forces[row] += f
                else:
                    right_forces[row] += f

        for ti, t in enumerate(frame_times):
            lf = np.zeros(3, dtype=np.float64)
            rf = np.zeros(3, dtype=np.float64)
            for comp in range(3):
                lf[comp] = float(np.interp(float(t), sol_times, left_forces[:, comp]))
                rf[comp] = float(np.interp(float(t), sol_times, right_forces[:, comp]))
            grf[ti, 0:3] = lf
            grf[ti, 6:9] = rf
            grf[ti, 12] = lf[1] + rf[1]
            grf[ti, 13] = float(np.linalg.norm(lf))
            grf[ti, 14] = float(np.linalg.norm(rf))
            grf[ti, 17] = 1.0
    except Exception:
        pass
    return grf.astype(np.float32)


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

    grf = np.full((t_len, SIM_GRF_COLS), np.nan, dtype=np.float32)
    moco_sol: osim.MocoSolution | None = None
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
        _, reserve_meta = _analyze_moco_reserve_controls(
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
        if solve_ok:
            grf = _extract_sim_grf_from_moco_solution(
                moco_sol,
                moco_model_path,
                frame_times,
                _track_config(cfg),
            )
    except Exception as exc:
        activations = np.full((t_len, len(muscle_name_list)), np.nan, dtype=np.float32)
        solve_ok = False
        solve_meta["success"] = False
        solve_meta["error"] = str(exc)

    return activations, solve_ok, solve_meta, out_sto, grf


def run_moco_track(
    q: np.ndarray,
    *,
    cfg: MuscleActivationConfig,
    work_dir: Path,
) -> MuscleActivationResult:
    """Run segmented MocoTrack and return activations ``[T, M]``."""
    _sweep_repo_root_moco_artifacts()
    arr = np.asarray(q, dtype=np.float64)
    t_len = int(arr.shape[0])
    if t_len < 2:
        raise ValueError(f"Need at least 2 frames for MocoTrack, got {t_len}")

    from nimble.moco_segment import run_moco_track_segmented
    from nimble.physics import load_model

    sk = load_model().skeleton
    return run_moco_track_segmented(
        arr,
        cfg=cfg,
        work_dir=work_dir,
        skeleton=sk,
    )
