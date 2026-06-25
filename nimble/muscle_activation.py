"""OpenSim muscle activations from Nimble Rajagopal ``q``.

Supports three methods: skip (``none``), ``moco_track``, and
``static_optimization``. Results are cached in B3D as ``muscle_activations``
``[M, T]``. Non-finite frames are linearly interpolated before optional
temporal smoothing.

OpenSim solves are **not differentiable**; use cached labels for training.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import opensim as osim

from nimble.rajagopal_coord_map import RAJAGOPAL_NIMBLE_DOF_NAMES

ACTIVATION_METHODS: Tuple[str, ...] = ("none", "moco_track", "static_optimization")


def rajagopal_model_path() -> Path:
    """Path to the bundled Rajagopal 2015 ``.osim`` (same as nimblephysics)."""
    import nimblephysics as nimble

    return Path(nimble.__file__).parent / "models" / "rajagopal_data" / "Rajagopal2015.osim"


@dataclass
class MuscleActivationConfig:
    """Configuration for OpenSim muscle activation estimation."""

    activation_method: str = "moco_track"
    fps: float = 20.0
    mass_kg: float = 70.0
    interpolate_activations: bool = True
    activation_smooth_hz: float = 6.0
    opensim_log_level: str = "Off"
    temp_dir: Optional[str] = None
    keep_temp: bool = False
    # MocoTrack defaults (OpenCap-like 0.05 s mesh at 20 fps, adaptive to 0.02 s).
    mesh_interval: Optional[float] = 0.05
    moco_residual_force: float = 250.0
    moco_reserve_optimal_force: float = 250.0
    moco_reserve_scale: float = 1.3
    moco_convergence_tolerance: float = 0.01
    # -1: OpenSim/Ipopt default (3000); set >0 to cap iterations explicitly.
    moco_max_iterations: int = -1
    moco_states_tracking_weight: float = 1.0
    moco_states_speed_tracking_weight: float = 1.0
    moco_aux_coord_tracking_weight: float = 1.0
    moco_control_effort_weight: float = 0.1
    moco_reference_lowpass_hz: float = 6.0
    moco_apply_tracked_states_to_guess: bool = True
    moco_minimize_implicit_aux_derivatives: bool = True
    moco_implicit_aux_derivatives_weight: float = 1e-6
    moco_reserve_control_weight: float = 0.001
    moco_weld_toe_joints: bool = True
    moco_contact_sphere_radius_m: float = 0.03
    moco_contact_toe_radius_m: float = 0.015
    moco_contact_sphere_offset_y_m: float = -0.02
    moco_contact_stiffness: float = 1e5
    moco_contact_dissipation: float = 200.0
    moco_multi_contact: bool = True
    moco_adaptive_mesh: bool = True
    moco_adaptive_mesh_speed_deg_s: float = 140.0
    moco_adaptive_mesh_interval: float = 0.02
    moco_max_reserve_fraction: float = 0.10
    moco_fail_on_high_reserve: bool = False
    moco_min_frames: int = 20
    moco_min_ik_success_fraction: float = 0.5
    moco_max_mean_fk_loss: Optional[float] = None
    moco_max_pelvis_ty_range_m: float = 0.8
    # Static optimization (OpenSim AnalyzeTool + StaticOptimization)
    static_activation_exponent: float = 2.0
    static_convergence_criterion: float = 1e-4
    static_max_iterations: int = 100
    static_lowpass_cutoff_hz: float = 6.0


def normalize_activation_method(method: str) -> str:
    key = str(method).strip().lower()
    aliases = {
        "skip": "none",
        "moco": "moco_track",
        "static": "static_optimization",
        "static_opt": "static_optimization",
    }
    key = aliases.get(key, key)
    if key not in ACTIVATION_METHODS:
        raise ValueError(
            f"Unknown activation_method {method!r}; "
            f"expected one of {ACTIVATION_METHODS}"
        )
    return key


def resolve_activation_method(args: argparse.Namespace) -> str:
    """CLI: ``--skip_muscle_activation`` maps to ``none``; else ``--activation_method``."""
    if bool(getattr(args, "skip_muscle_activation", False)):
        return "none"
    raw = getattr(args, "activation_method", None)
    if raw is None:
        return "moco_track"
    return normalize_activation_method(str(raw))


def muscle_activation_config_from_dict(data: Dict[str, Any]) -> MuscleActivationConfig:
    """Build config from a JSON-friendly dict (e.g. preprocess worker payload)."""
    d = dict(data)
    fields = {
        k: v for k, v in d.items() if k in MuscleActivationConfig.__dataclass_fields__
    }
    if "activation_method" in fields:
        fields["activation_method"] = normalize_activation_method(
            str(fields["activation_method"])
        )
    return MuscleActivationConfig(**fields)


def muscle_activation_config_to_dict(cfg: MuscleActivationConfig) -> Dict[str, Any]:
    """Serialize config for multiprocessing / manifest metadata."""
    return asdict(cfg)


def add_muscle_activation_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register muscle activation CLI flags shared by preprocess and offline scripts."""
    grp = parser.add_argument_group("muscle activation")
    grp.add_argument(
        "--activation_method",
        choices=ACTIVATION_METHODS,
        default="moco_track",
        help=(
            "Muscle label method: none (skip), moco_track (default), "
            "or static_optimization (faster, lower fidelity)."
        ),
    )
    static = parser.add_argument_group("static optimization")
    static.add_argument("--static_activation_exponent", type=float, default=None)
    static.add_argument("--static_convergence_criterion", type=float, default=None)
    static.add_argument("--static_max_iterations", type=int, default=None)
    static.add_argument("--static_lowpass_cutoff_hz", type=float, default=None)
    grp = parser.add_argument_group("moco track")
    grp.add_argument(
        "--moco_mesh_interval",
        type=float,
        default=None,
        help="Moco mesh interval in seconds (default 0.05, matches OpenCap Moco).",
    )
    grp.add_argument("--moco_residual_force", type=float, default=None)
    grp.add_argument(
        "--moco_reserve_optimal_force",
        type=float,
        default=None,
        help="Reserve actuator optimal force in N (default 250, OpenSim walking example).",
    )
    grp.add_argument(
        "--moco_reserve_scale",
        type=float,
        default=None,
        help="Multiplier on reserve optimal force (default 1.3, +30%% athletic tuning).",
    )
    grp.add_argument(
        "--moco_reserve_control_weight",
        type=float,
        default=None,
        help="MocoControlGoal weight for reserve actuators (default 0.001).",
    )
    grp.add_argument("--moco_convergence_tolerance", type=float, default=None)
    grp.add_argument(
        "--moco_max_iterations",
        type=int,
        default=None,
        help=(
            "Ipopt iteration cap (>0). Default -1 uses the solver default (3000); "
            "matches OpenCap/OpenSim MocoTrack when unset."
        ),
    )
    grp.add_argument("--moco_states_tracking_weight", type=float, default=None)
    grp.add_argument(
        "--moco_states_speed_tracking_weight",
        type=float,
        default=None,
        help="Moco tracking weight for joint /speed states (default 1.0; pelvis_ty/value stays 0).",
    )
    grp.add_argument(
        "--moco_aux_coord_tracking_weight",
        type=float,
        default=None,
        help="Unused (kept for CLI compat); position tracking uses --moco_states_tracking_weight.",
    )
    grp.add_argument(
        "--moco_reference_lowpass_hz",
        type=float,
        default=None,
        help="Low-pass cutoff (Hz) on Moco reference coordinates (default 6; 0 disables).",
    )
    grp.add_argument(
        "--moco_no_reference_lowpass",
        action="store_true",
        help="Disable low-pass filtering of the Moco reference table.",
    )
    grp.add_argument(
        "--moco_no_apply_tracked_guess",
        action="store_true",
        help="Do not seed Moco from the tracked reference when no warm-start guess exists.",
    )
    grp.add_argument(
        "--moco_no_implicit_aux_derivatives",
        action="store_true",
        help="Disable OpenSim Moco implicit muscle auxiliary derivative minimization.",
    )
    grp.add_argument(
        "--moco_no_weld_toes",
        action="store_true",
        help="Do not weld MTP joints in MocoTrack.",
    )
    grp.add_argument(
        "--moco_no_multi_contact",
        action="store_true",
        help="Use single calcaneus sphere per foot instead of calcaneus+toe.",
    )
    grp.add_argument(
        "--moco_no_adaptive_mesh",
        action="store_true",
        help="Disable finer mesh for high joint-speed motion.",
    )
    grp.add_argument(
        "--moco_adaptive_mesh_speed_deg_s",
        type=float,
        default=None,
        help="Joint-speed threshold (deg/s) for adaptive mesh (default 140).",
    )
    grp.add_argument(
        "--moco_adaptive_mesh_interval",
        type=float,
        default=None,
        help="Mesh interval when adaptive mesh triggers (default 0.01 s).",
    )
    grp.add_argument(
        "--moco_max_reserve_fraction",
        type=float,
        default=None,
        help="Max reserve control magnitude before QC fail (default 0.10).",
    )
    grp.add_argument(
        "--moco_allow_high_reserve",
        action="store_true",
        help="Keep frames valid even when reserve actuators exceed QC threshold.",
    )
    grp.add_argument(
        "--moco_contact_toe_radius_m",
        type=float,
        default=None,
        help="Toe contact sphere radius in m (default 0.015).",
    )
    grp.add_argument(
        "--moco_min_frames",
        type=int,
        default=20,
        help="Skip Moco when clip has fewer frames.",
    )
    grp.add_argument(
        "--moco_min_ik_success_fraction",
        type=float,
        default=0.5,
        help="Minimum IK success_ratio before running Moco (0 disables).",
    )
    grp.add_argument(
        "--moco_max_mean_fk_loss",
        type=float,
        default=-1.0,
        help="Max mean_fk_loss for Moco; <=0 disables.",
    )
    grp.add_argument(
        "--moco_max_pelvis_ty_range_m",
        type=float,
        default=None,
        help="Reject clip when pelvis vertical range exceeds this (m); default 0.8.",
    )
    grp.add_argument(
        "--moco_no_repair",
        action="store_true",
        help="Disable temporal interpolation of non-finite activation frames.",
    )
    grp.add_argument(
        "--activation_smooth_hz",
        type=float,
        default=None,
        help=(
            "Low-pass muscle activations along time after repair (Hz); "
            "0 disables. Default: 6."
        ),
    )
    grp.add_argument(
        "--opensim_log_level",
        default="Off",
        choices=("Off", "Critical", "Error", "Warn", "Info", "Debug"),
        help=(
            "OpenSim log verbosity during Moco/IK (default Off). "
            "Off also suppresses Rajagopal mesh warnings on the terminal."
        ),
    )


def _fail_on_high_reserve_from_args(args: argparse.Namespace) -> bool:
    """Reserve QC defaults: on for MocoTrack, off for static optimization."""
    if bool(getattr(args, "moco_allow_high_reserve", False)):
        return False
    if resolve_activation_method(args) == "static_optimization":
        return False
    return True


def muscle_activation_config_from_args(
    args: argparse.Namespace,
    *,
    fps: float | None = None,
    mass_kg: float | None = None,
    keep_temp: bool = False,
) -> MuscleActivationConfig:
    """Build ``MuscleActivationConfig`` from CLI namespace."""
    base = MuscleActivationConfig()

    def _pick(attr: str, arg_name: str, *, cast=float):
        val = getattr(args, arg_name, None)
        return cast(val) if val is not None else getattr(base, attr)

    max_fk = float(getattr(args, "moco_max_mean_fk_loss", -1.0))
    mesh = getattr(args, "moco_mesh_interval", None)
    return MuscleActivationConfig(
        activation_method=resolve_activation_method(args),
        fps=float(fps if fps is not None else getattr(args, "fps", base.fps)),
        mass_kg=float(
            mass_kg if mass_kg is not None else getattr(args, "mass_kg", base.mass_kg)
        ),
        mesh_interval=float(mesh) if mesh is not None else base.mesh_interval,
        moco_residual_force=_pick("moco_residual_force", "moco_residual_force"),
        moco_reserve_optimal_force=_pick(
            "moco_reserve_optimal_force", "moco_reserve_optimal_force"
        ),
        moco_reserve_scale=_pick("moco_reserve_scale", "moco_reserve_scale"),
        moco_reserve_control_weight=_pick(
            "moco_reserve_control_weight", "moco_reserve_control_weight"
        ),
        moco_convergence_tolerance=_pick(
            "moco_convergence_tolerance", "moco_convergence_tolerance"
        ),
        moco_max_iterations=int(
            _pick("moco_max_iterations", "moco_max_iterations", cast=int)
        ),
        moco_states_tracking_weight=_pick(
            "moco_states_tracking_weight", "moco_states_tracking_weight"
        ),
        moco_states_speed_tracking_weight=_pick(
            "moco_states_speed_tracking_weight", "moco_states_speed_tracking_weight"
        ),
        moco_aux_coord_tracking_weight=_pick(
            "moco_aux_coord_tracking_weight", "moco_aux_coord_tracking_weight"
        ),
        moco_reference_lowpass_hz=(
            0.0
            if bool(getattr(args, "moco_no_reference_lowpass", False))
            else float(
                getattr(args, "moco_reference_lowpass_hz", None)
                if getattr(args, "moco_reference_lowpass_hz", None) is not None
                else base.moco_reference_lowpass_hz
            )
        ),
        moco_apply_tracked_states_to_guess=not bool(
            getattr(args, "moco_no_apply_tracked_guess", False)
        ),
        moco_minimize_implicit_aux_derivatives=not bool(
            getattr(args, "moco_no_implicit_aux_derivatives", False)
        ),
        moco_weld_toe_joints=not bool(getattr(args, "moco_no_weld_toes", False)),
        moco_multi_contact=not bool(getattr(args, "moco_no_multi_contact", False)),
        moco_adaptive_mesh=not bool(getattr(args, "moco_no_adaptive_mesh", False)),
        moco_adaptive_mesh_speed_deg_s=_pick(
            "moco_adaptive_mesh_speed_deg_s", "moco_adaptive_mesh_speed_deg_s"
        ),
        moco_adaptive_mesh_interval=_pick(
            "moco_adaptive_mesh_interval", "moco_adaptive_mesh_interval"
        ),
        moco_max_reserve_fraction=_pick(
            "moco_max_reserve_fraction", "moco_max_reserve_fraction"
        ),
        moco_fail_on_high_reserve=_fail_on_high_reserve_from_args(args),
        moco_contact_toe_radius_m=_pick(
            "moco_contact_toe_radius_m", "moco_contact_toe_radius_m"
        ),
        moco_min_frames=int(getattr(args, "moco_min_frames", base.moco_min_frames)),
        moco_min_ik_success_fraction=float(
            getattr(
                args,
                "moco_min_ik_success_fraction",
                base.moco_min_ik_success_fraction,
            )
        ),
        moco_max_mean_fk_loss=(
            float(max_fk) if max_fk > 0.0 else base.moco_max_mean_fk_loss
        ),
        moco_max_pelvis_ty_range_m=_pick(
            "moco_max_pelvis_ty_range_m", "moco_max_pelvis_ty_range_m"
        ),
        interpolate_activations=not bool(getattr(args, "moco_no_repair", False)),
        activation_smooth_hz=float(
            getattr(args, "activation_smooth_hz", None)
            if getattr(args, "activation_smooth_hz", None) is not None
            else base.activation_smooth_hz
        ),
        opensim_log_level=str(
            getattr(args, "opensim_log_level", base.opensim_log_level)
        ),
        keep_temp=bool(keep_temp),
        static_activation_exponent=float(
            getattr(args, "static_activation_exponent", None)
            if getattr(args, "static_activation_exponent", None) is not None
            else base.static_activation_exponent
        ),
        static_convergence_criterion=float(
            getattr(args, "static_convergence_criterion", None)
            if getattr(args, "static_convergence_criterion", None) is not None
            else base.static_convergence_criterion
        ),
        static_max_iterations=int(
            getattr(args, "static_max_iterations", None)
            if getattr(args, "static_max_iterations", None) is not None
            else base.static_max_iterations
        ),
        static_lowpass_cutoff_hz=float(
            getattr(args, "static_lowpass_cutoff_hz", None)
            if getattr(args, "static_lowpass_cutoff_hz", None) is not None
            else base.static_lowpass_cutoff_hz
        ),
    )


def _normalize_opensim_log_level(level: str) -> str:
    key = str(level).strip().lower()
    if key in ("", "silent", "none", "quiet"):
        return "Off"
    return str(level).strip()


def configure_opensim_logging(level: str = "Off") -> None:
    """Set global OpenSim logger level (call early in worker / before loading models)."""
    try:
        osim.Logger.setLevelString(_normalize_opensim_log_level(level))
    except Exception:
        pass


def _opensim_stdio_suppressed(level: str) -> bool:
    return _normalize_opensim_log_level(level) == "Off"


@contextmanager
def _suppress_process_stdio():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        os.close(devnull_fd)


@contextmanager
def opensim_quiet(level: str = "Off"):
    """Suppress OpenSim Logger + console during Moco solves."""
    normalized = _normalize_opensim_log_level(level)
    prev = osim.Logger.getLevelString()
    configure_opensim_logging(normalized)
    if _opensim_stdio_suppressed(normalized):
        with _suppress_process_stdio():
            try:
                yield
            finally:
                try:
                    osim.Logger.setLevelString(prev)
                except Exception:
                    pass
    else:
        try:
            yield
        finally:
            try:
                osim.Logger.setLevelString(prev)
            except Exception:
                pass


@dataclass
class MuscleActivationResult:
    """Muscle activation output for one motion."""

    activations: np.ndarray
    muscle_names: Tuple[str, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)
    forces: Optional[np.ndarray] = None

    @property
    def num_frames(self) -> int:
        return int(self.activations.shape[0])

    @property
    def num_muscles(self) -> int:
        return int(self.activations.shape[1])


def muscle_names(model: Any | None = None) -> Tuple[str, ...]:
    """Ordered muscle names from the Rajagopal model (80 muscles)."""
    if model is None:
        with opensim_quiet("Off"):
            model = osim.Model(str(rajagopal_model_path()))
            model.initSystem()
    muscles = model.getMuscles()
    return tuple(muscles.get(i).getName() for i in range(muscles.getSize()))


def interpolate_activation_frames(
    activations: np.ndarray,
) -> tuple[np.ndarray, Dict[str, int]]:
    """Linearly interpolate non-finite activation timesteps from finite neighbors."""
    act = np.asarray(activations, dtype=np.float32).copy()
    t_len = int(act.shape[0])
    meta = {
        "repaired_frame_count": 0,
        "interpolated_frame_count": 0,
        "extrapolated_frame_count": 0,
    }
    if t_len == 0:
        return act, meta

    good = np.isfinite(act).all(axis=1)
    need_repair = ~good
    meta["repaired_frame_count"] = int(need_repair.sum())
    for t in np.where(need_repair)[0]:
        left: int | None = None
        for i in range(t - 1, -1, -1):
            if good[i]:
                left = i
                break
        right: int | None = None
        for i in range(t + 1, t_len):
            if good[i]:
                right = i
                break
        if left is not None and right is not None:
            w = float(t - left) / float(right - left)
            act[t] = (1.0 - w) * act[left] + w * act[right]
            meta["interpolated_frame_count"] += 1
        elif left is not None:
            act[t] = act[left]
            meta["extrapolated_frame_count"] += 1
        elif right is not None:
            act[t] = act[right]
            meta["extrapolated_frame_count"] += 1
        else:
            act[t] = 0.0

    return act, meta


def apply_activation_postprocess(
    activations: np.ndarray,
    cfg: MuscleActivationConfig,
    *,
    force_repair: bool = False,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Interpolate non-finite frames and optionally low-pass the activation trajectory."""
    from nimble.smoothing import smooth_activation_trajectory

    meta: Dict[str, Any] = {}
    act = np.asarray(activations, dtype=np.float32)
    needs_repair = bool(force_repair) or bool(cfg.interpolate_activations)
    if not np.isfinite(act).all():
        needs_repair = True

    if needs_repair:
        act, repair_meta = interpolate_activation_frames(act)
        meta.update(repair_meta)

    if float(cfg.activation_smooth_hz) > 0.0:
        act = smooth_activation_trajectory(
            act,
            fps=float(cfg.fps),
            cutoff_hz=float(cfg.activation_smooth_hz),
        )
        meta["activation_smooth_hz"] = float(cfg.activation_smooth_hz)

    if not np.isfinite(act).all():
        act = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return act, meta


def _storage_to_array(storage: Any) -> Tuple[np.ndarray, List[str]]:
    labels: List[str] = []
    for i in range(storage.getColumnLabels().size()):
        labels.append(storage.getColumnLabels().get(i))
    rows: List[List[float]] = []
    for i in range(storage.getSize()):
        sv = storage.getStateVector(i)
        d = sv.getData()
        rows.append([float(d.get(j)) for j in range(d.size())])
    return np.asarray(rows, dtype=np.float64), labels


def _validate_q_input(q: np.ndarray) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected q [T, ndof], got {arr.shape}")
    if arr.shape[1] != len(RAJAGOPAL_NIMBLE_DOF_NAMES):
        raise ValueError(
            f"Expected q [T, {len(RAJAGOPAL_NIMBLE_DOF_NAMES)}], got {arr.shape}"
        )
    return arr


def _activation_work_dir(cfg: MuscleActivationConfig, *, prefix: str) -> Tuple[Path, bool]:
    if cfg.temp_dir:
        work_dir = Path(cfg.temp_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir, False
    work_dir = Path(tempfile.mkdtemp(prefix=prefix))
    return work_dir, not bool(cfg.keep_temp)


def _finalize_activation_result(
    result: MuscleActivationResult,
    cfg: MuscleActivationConfig,
    *,
    method_label: str,
) -> MuscleActivationResult:
    had_non_finite = not np.isfinite(result.activations).all()
    activations, repair_meta = apply_activation_postprocess(
        result.activations,
        cfg,
        force_repair=had_non_finite,
    )
    metadata = dict(result.metadata)
    metadata.update(repair_meta)
    return MuscleActivationResult(
        activations=activations,
        muscle_names=result.muscle_names,
        metadata=metadata,
        forces=result.forces,
    )


def fallback_muscle_activations(
    num_frames: int,
    cfg: MuscleActivationConfig,
    *,
    num_muscles: int | None = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Build a repaired activation trajectory when OpenSim muscle solve fails entirely."""
    n_muscles = int(num_muscles or len(muscle_names()))
    placeholder = np.full((int(num_frames), n_muscles), np.nan, dtype=np.float32)
    activations, meta = apply_activation_postprocess(placeholder, cfg, force_repair=True)
    return activations, meta


def compute_muscle_activation(
    q: np.ndarray,
    *,
    cfg: MuscleActivationConfig | None = None,
) -> MuscleActivationResult:
    """Compute full-body muscle activations from Nimble Rajagopal ``q`` ``[T, 37]``."""
    cfg = cfg or MuscleActivationConfig()
    method = normalize_activation_method(cfg.activation_method)
    if method == "none":
        raise ValueError(
            "activation_method is 'none'; export with skip before calling compute_muscle_activation"
        )

    configure_opensim_logging(cfg.opensim_log_level)
    arr = _validate_q_input(q)

    prefix = (
        "sindyffuse_static_"
        if method == "static_optimization"
        else "sindyffuse_moco_"
    )
    work_dir, cleanup = _activation_work_dir(cfg, prefix=prefix)

    try:
        if method == "static_optimization":
            from nimble.static_optimization import run_static_optimization

            result = run_static_optimization(arr, cfg=cfg, work_dir=work_dir)
            label = "Static optimization"
        else:
            from nimble.moco_track import run_moco_track

            result = run_moco_track(arr, cfg=cfg, work_dir=work_dir)
            label = "MocoTrack"
        return _finalize_activation_result(result, cfg, method_label=label)
    finally:
        if cleanup:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)


def activation_stats(activations: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-frame scalar summaries for guidance penalties."""
    act = np.asarray(activations, dtype=np.float32)
    if act.ndim != 2:
        raise ValueError(f"Expected activations [T, M], got {act.shape}")
    mean_activation = np.nanmean(act, axis=1)
    max_activation = np.nanmax(act, axis=1)
    if act.shape[0] > 1:
        diff = np.diff(act, axis=0)
        smooth_tail = np.linalg.norm(diff, axis=1)
        activation_smoothness = np.concatenate(
            [np.asarray([0.0], dtype=np.float32), smooth_tail.astype(np.float32)]
        )
    else:
        activation_smoothness = np.zeros((1,), dtype=np.float32)
    return {
        "mean_activation": mean_activation.astype(np.float32),
        "max_activation": max_activation.astype(np.float32),
        "activation_smoothness": activation_smoothness.astype(np.float32),
    }
