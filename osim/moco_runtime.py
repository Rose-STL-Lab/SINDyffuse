"""
MocoTrack-based biomechanics: torque-driven marker tracking → dynamics-consistent kinematics +
generalized coordinate forces via :meth:`MocoStudy.calcGeneralizedForces`.

Requires OpenSim Python **with Moco** (conda ``opensim`` / ``opensim-moco`` builds).

Used by ``guidance`` (numpy oracle) and ``pipeline`` (always merges Moco features into trials).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

import opensim as osim  # type: ignore[import-untyped]

from common.paths import DEFAULT_MODEL_PATH
from osim.rajagopal_markers import DEFAULT_HML_MARKER_PAIRS, marker_names_missing
from osim.smoothing import apply_pose_smoothing


def _independent_column_numpy(table: Any) -> np.ndarray:
    col = table.getIndependentColumn()
    if hasattr(col, "to_numpy"):
        return np.asarray(col.to_numpy(), dtype=np.float64).reshape(-1)
    gs = getattr(col, "getSize", None)
    if callable(gs):
        n = int(gs())
        return np.array([float(col[i]) for i in range(n)], dtype=np.float64)
    if callable(getattr(col, "size", None)):
        n = int(col.size())  # type: ignore[misc]
        return np.array([float(col[i]) for i in range(n)], dtype=np.float64)
    return np.asarray(col, dtype=np.float64).reshape(-1)


def _build_marker_timeseries_vec3(
    poses: np.ndarray,
    dt: float,
    marker_pairs: Sequence[Tuple[int, str]],
) -> osim.TimeSeriesTableVec3:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (22, 3):
        raise ValueError(f"Expected poses [T,22,3], got {poses.shape}")

    pairs = list(marker_pairs)
    labels = [m for _, m in pairs]

    tab = osim.TimeSeriesTableVec3()
    tab.setColumnLabels(list(labels))

    for ti in range(poses.shape[0]):
        row = osim.RowVectorVec3(
            [
                osim.Vec3(
                    float(poses[ti, ji, 0]),
                    float(poses[ti, ji, 1]),
                    float(poses[ti, ji, 2]),
                )
                for ji, _ in pairs
            ]
        )
        tab.appendRow(float(ti) * float(dt), row)
    return tab


def _state_speed_proxy_l2(table: Any) -> np.ndarray:
    """Row-wise L2 norm over */speed columns in a generic TimeSeriesTable."""
    labs = [str(x) for x in table.getColumnLabels() if str(x).endswith("/speed")]
    if not labs:
        return np.zeros(int(table.getNumRows()), dtype=np.float64)
    mats = []
    for lab in labs:
        mats.append(table.getDependentColumn(lab).to_numpy())
    M = np.stack(mats, axis=1).astype(np.float64)
    return np.linalg.norm(M, axis=1)


def moco_marker_track_feature_summary(
    poses: np.ndarray,
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    dt: float,
    fps: float,
    max_frames: int,
    ik_marker_pairs: Sequence[Tuple[int, str]] | None = None,
    smooth_before_track: bool = True,
    smooth_cutoff_hz: float = 6.0,
    smooth_butterworth_order: int = 2,
    weld_toes: bool = True,
    markers_global_tracking_weight: float = 10.0,
    control_effort_weight: float = 0.1,
    mesh_interval: float | None = None,
    markers_lowpass_hz: float = 0.0,
    max_solver_iterations: int = 200,
    convergence_tolerance: float = 5e-2,
    constraint_tolerance: float = 5e-2,
) -> Dict[str, Any]:
    """
    Run MocoTrack (torque-driven, reserves) with marker references built from poses.

    Returns per-frame summaries compatible with the guidance oracle aggregator.
    """
    poses_arr = np.asarray(poses)
    mf = max(2, int(max_frames))
    poses_used = poses_arr[:mf] if poses_arr.shape[0] > mf else poses_arr
    if poses_used.shape[0] < 2:
        raise ValueError("Moco marker tracking needs at least two time samples.")

    pose_work = poses_used.astype(np.float64)
    if smooth_before_track:
        pose_work32, _ = apply_pose_smoothing(
            poses_used.astype(np.float32),
            fps=int(round(fps)),
            sampling_frequency=None,
            smooth_poses=True,
            smooth_cutoff_hz=float(smooth_cutoff_hz),
            smooth_butterworth_order=int(smooth_butterworth_order),
        )
        pose_work = pose_work32.astype(np.float64)

    pairs = list(ik_marker_pairs) if ik_marker_pairs is not None else list(DEFAULT_HML_MARKER_PAIRS)
    mp = Path(model_path)
    miss = marker_names_missing(mp, [m for _, m in pairs])
    if miss:
        raise ValueError(f"Model lacks markers needed for marker reference: {miss}")

    markers_vec = _build_marker_timeseries_vec3(pose_work, float(dt), pairs)
    markers_flat = markers_vec.flatten()
    proc = osim.TableProcessor(markers_flat)
    if float(markers_lowpass_hz) > 0:
        proc.append(osim.TabOpLowPassFilter(float(markers_lowpass_hz)))

    final_t = float(pose_work.shape[0] - 1) * float(dt)
    mh = float(mesh_interval if mesh_interval is not None else min(0.05, max(float(dt), 2e-3)))

    model_processor = osim.ModelProcessor(str(mp))
    if weld_toes:
        weld = osim.StdVectorString()
        weld.append("mtp_r")
        weld.append("mtp_l")
        model_processor.append(osim.ModOpReplaceJointsWithWelds(weld))
    model_processor.append(osim.ModOpRemoveMuscles())
    model_processor.append(osim.ModOpAddResiduals(250.0, 50.0, 1.0))
    model_processor.append(osim.ModOpAddReserves(250.0, 1.0))

    track = osim.MocoTrack()
    track.setName("hml3d_moco_marker_track_torque_reserves")
    track.setModel(model_processor)
    track.setMarkersReference(proc)
    track.set_allow_unused_references(True)
    track.set_markers_global_tracking_weight(float(markers_global_tracking_weight))
    track.set_control_effort_weight(float(control_effort_weight))
    track.set_initial_time(0.0)
    track.set_final_time(final_t)
    track.set_mesh_interval(mh)

    study = track.initialize()
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
    solver.set_optim_max_iterations(int(max_solver_iterations))
    solver.set_optim_convergence_tolerance(float(convergence_tolerance))
    solver.set_optim_constraint_tolerance(float(constraint_tolerance))
    solver.resetProblem(study.updProblem())

    solution = study.solve()
    if hasattr(solution, "success") and not solution.success():
        raise RuntimeError(f"MocoTrack solve failed: {solution.getStatus()}")

    gf_table = study.calcGeneralizedForces(solution, ["/forceset/.*"])
    gcols = [str(x) for x in gf_table.getColumnLabels()]
    gmat = np.stack([gf_table.getDependentColumn(c).to_numpy().astype(np.float64) for c in gcols], axis=1)
    tau_mesh = np.linalg.norm(gmat, axis=1)

    ctrl = solution.getControlsTrajectoryMat()
    assert ctrl is not None
    ctrln = np.asarray(ctrl, dtype=np.float64)
    effort_mesh = np.sum(np.abs(ctrln), axis=1)

    states_table = solution.exportToStatesTable()
    spd_mesh = _state_speed_proxy_l2(states_table)

    # Optimization mesh differs from offline marker samples; interpolate to `[0..T_src-1]*dt`.
    t_tgt = np.arange(pose_work.shape[0], dtype=np.float64) * float(dt)
    t_gf = _independent_column_numpy(gf_table)
    tau_series = np.interp(t_tgt, t_gf, tau_mesh.astype(np.float64)).astype(np.float32)
    effort_series = np.interp(t_tgt, t_gf, effort_mesh.astype(np.float64)).astype(np.float32)
    ts_s = _independent_column_numpy(states_table)
    spd_series = np.interp(t_tgt, ts_s, spd_mesh.astype(np.float64))
    acc_proxy = np.abs(np.gradient(spd_series, float(dt))).astype(np.float32)
    jerk_proxy = np.abs(np.gradient(acc_proxy.astype(np.float64), float(dt))).astype(np.float32)

    return {
        "moco_torque_l2": tau_series,
        "moco_vel_proxy_l2": spd_series.astype(np.float32),
        "moco_acc_proxy_l2": acc_proxy,
        "moco_jerk_proxy_l2": jerk_proxy,
        "moco_effort_l1": effort_series,
        "moco_frames_used": np.array([float(pose_work.shape[0])], dtype=np.float32),
    }
