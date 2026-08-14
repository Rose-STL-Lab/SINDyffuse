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
ACTIVATION_METHODS: Tuple[str, ...] = ('opensimad', 'moco_track')

def rajagopal_model_path() -> Path:
    import nimblephysics as nimble
    return Path(nimble.__file__).parent / 'models' / 'rajagopal_data' / 'Rajagopal2015.osim'

@dataclass
class MuscleActivationConfig:
    activation_method: str = 'opensimad'
    fps: float = 20.0
    mass_kg: float = 70.0
    opensim_log_level: str = 'Off'
    temp_dir: Optional[str] = None
    keep_temp: bool = False
    mesh_interval: Optional[float] = 0.02
    moco_residual_force: float = 250.0
    moco_reserve_optimal_force: float = 250.0
    moco_reserve_scale: float = 1.3
    moco_convergence_tolerance: float = 0.001
    moco_max_iterations: int = 2500
    moco_states_tracking_weight: float = 1.0
    moco_states_speed_tracking_weight: float = 1.0
    moco_aux_coord_tracking_weight: float = 1.0
    moco_control_effort_weight: float = 0.1
    moco_reference_lowpass_hz: float = 6.0
    moco_apply_tracked_states_to_guess: bool = True
    moco_minimize_implicit_aux_derivatives: bool = True
    moco_implicit_aux_derivatives_weight: float = 1e-06
    moco_reserve_control_weight: float = 0.001
    moco_weld_toe_joints: bool = True
    moco_contact_sphere_radius_m: float = 0.03
    moco_contact_toe_radius_m: float = 0.015
    moco_contact_sphere_offset_y_m: float = -0.02
    moco_contact_stiffness: float = 100000.0
    moco_contact_dissipation: float = 200.0
    moco_multi_contact: bool = True
    moco_adaptive_mesh: bool = False
    moco_adaptive_mesh_speed_deg_s: float = 140.0
    moco_adaptive_mesh_interval: float = 0.02
    moco_use_function_based_paths: bool = True
    moco_parallel_segments: int = 6
    moco_core_duration_s: float = 1.4
    moco_buffer_duration_s: float = 0.14
    moco_stitch_blend_s: float = 0.14

def normalize_activation_method(method: str) -> str:
    key = str(method).strip().lower()
    aliases = {
        'moco': 'moco_track',
        'ik': 'opensimad',
        'opensim_ad': 'opensimad',
        'open_sim_ad': 'opensimad',
        'mint': 'opensimad',
    }
    key = aliases.get(key, key)
    if key not in ACTIVATION_METHODS:
        raise ValueError(f'Unknown activation_method {method!r}; expected one of {ACTIVATION_METHODS}')
    return key

def resolve_activation_method(args: argparse.Namespace) -> str:
    if bool(getattr(args, 'skip_muscle_activation', False)):
        raise ValueError('skip_muscle_activation is not supported; use preprocess_ik.py for IK-only export')
    raw = getattr(args, 'activation_method', None)
    if raw is None:
        return 'opensimad'
    return normalize_activation_method(str(raw))

def muscle_activation_config_from_dict(data: Dict[str, Any]) -> MuscleActivationConfig:
    d = dict(data)
    fields = {k: v for k, v in d.items() if k in MuscleActivationConfig.__dataclass_fields__}
    if 'activation_method' in fields:
        fields['activation_method'] = normalize_activation_method(str(fields['activation_method']))
    return MuscleActivationConfig(**fields)

def muscle_activation_config_to_dict(cfg: MuscleActivationConfig) -> Dict[str, Any]:
    return asdict(cfg)

def add_muscle_activation_cli_args(parser: argparse.ArgumentParser) -> None:
    grp = parser.add_argument_group('muscle activation')
    grp.add_argument('--activation_method', choices=ACTIVATION_METHODS, default='opensimad', help='Muscle label method (default opensimad / MinT OpenSimAD; moco_track kept for A/B).')
    grp = parser.add_argument_group('moco track')
    grp.add_argument('--moco_mesh_interval', type=float, default=None, help='Mesh interval in seconds (default 0.02 = 50 colloc pts/s).')
    grp.add_argument('--moco_residual_force', type=float, default=None)
    grp.add_argument('--moco_reserve_optimal_force', type=float, default=None, help='Reserve actuator optimal force in N (default 250, OpenSim walking example).')
    grp.add_argument('--moco_reserve_scale', type=float, default=None, help='Multiplier on reserve optimal force (default 1.3, +30%% athletic tuning).')
    grp.add_argument('--moco_reserve_control_weight', type=float, default=None, help='MocoControlGoal weight for reserve actuators (default 0.001).')
    grp.add_argument('--moco_convergence_tolerance', type=float, default=None)
    grp.add_argument('--moco_max_iterations', type=int, default=None, help='Ipopt iteration cap (default 2500, MinT).')
    grp.add_argument('--moco_states_tracking_weight', type=float, default=None)
    grp.add_argument('--moco_states_speed_tracking_weight', type=float, default=None, help='Moco tracking weight for joint /speed states (default 1.0; pelvis_ty/value stays 0).')
    grp.add_argument('--moco_aux_coord_tracking_weight', type=float, default=None, help='Unused (kept for CLI compat); position tracking uses --moco_states_tracking_weight.')
    grp.add_argument('--moco_reference_lowpass_hz', type=float, default=None, help='Low-pass cutoff (Hz) on Moco reference coordinates (default 6; 0 disables).')
    grp.add_argument('--moco_no_reference_lowpass', action='store_true', help='Disable low-pass filtering of the Moco reference table.')
    grp.add_argument('--moco_no_apply_tracked_guess', action='store_true', help='Do not seed Moco from the tracked reference when no warm-start guess exists.')
    grp.add_argument('--moco_no_implicit_aux_derivatives', action='store_true', help='Disable OpenSim Moco implicit muscle auxiliary derivative minimization.')
    grp.add_argument('--moco_no_weld_toes', action='store_true', help='Do not weld MTP joints in MocoTrack.')
    grp.add_argument('--moco_no_multi_contact', action='store_true', help='Use single calcaneus sphere per foot instead of calcaneus+toe.')
    grp.add_argument('--moco_no_adaptive_mesh', action='store_true', help='Disable finer mesh for high joint-speed motion.')
    grp.add_argument('--moco_adaptive_mesh_speed_deg_s', type=float, default=None, help='Joint-speed threshold (deg/s) for adaptive mesh (default 140).')
    grp.add_argument('--moco_adaptive_mesh_interval', type=float, default=None, help='Mesh interval when adaptive mesh triggers (default 0.01 s).')
    grp.add_argument('--moco_contact_toe_radius_m', type=float, default=None, help='Toe contact sphere radius in m (default 0.015).')
    grp.add_argument('--moco_no_function_based_paths', action='store_true', help='Use geometry muscle paths instead of function-based paths.')
    grp.add_argument('--moco_parallel_segments', type=int, default=None, help='Concurrent segments per motion (default 6, MinT).')
    # --opensim_log_level lives on add_common_preprocess_args; avoid duplicate when both are used.
    if not any('--opensim_log_level' in getattr(action, 'option_strings', ()) for action in parser._actions):
        grp.add_argument('--opensim_log_level', default='Off', choices=('Off', 'Critical', 'Error', 'Warn', 'Info', 'Debug'), help='OpenSim log verbosity during Moco/IK (default Off). Off also suppresses Rajagopal mesh warnings on the terminal.')
    seg = parser.add_argument_group('segmented moco')
    seg.add_argument('--moco_core_duration_s', type=float, default=None)
    seg.add_argument('--moco_buffer_duration_s', type=float, default=None)
    seg.add_argument('--moco_stitch_blend_s', type=float, default=None)

def muscle_activation_config_from_args(args: argparse.Namespace, *, fps: float | None=None, mass_kg: float | None=None, keep_temp: bool=False) -> MuscleActivationConfig:
    base = MuscleActivationConfig()

    def _pick(attr: str, arg_name: str, *, cast=float):
        val = getattr(args, arg_name, None)
        return cast(val) if val is not None else getattr(base, attr)
    mesh = getattr(args, 'moco_mesh_interval', None)
    parallel_seg = getattr(args, 'moco_parallel_segments', None)
    return MuscleActivationConfig(activation_method=resolve_activation_method(args), fps=float(fps if fps is not None else getattr(args, 'fps', base.fps)), mass_kg=float(mass_kg if mass_kg is not None else getattr(args, 'mass_kg', base.mass_kg)), mesh_interval=float(mesh) if mesh is not None else base.mesh_interval, moco_residual_force=_pick('moco_residual_force', 'moco_residual_force'), moco_reserve_optimal_force=_pick('moco_reserve_optimal_force', 'moco_reserve_optimal_force'), moco_reserve_scale=_pick('moco_reserve_scale', 'moco_reserve_scale'), moco_reserve_control_weight=_pick('moco_reserve_control_weight', 'moco_reserve_control_weight'), moco_convergence_tolerance=_pick('moco_convergence_tolerance', 'moco_convergence_tolerance'), moco_max_iterations=int(_pick('moco_max_iterations', 'moco_max_iterations', cast=int)), moco_states_tracking_weight=_pick('moco_states_tracking_weight', 'moco_states_tracking_weight'), moco_states_speed_tracking_weight=_pick('moco_states_speed_tracking_weight', 'moco_states_speed_tracking_weight'), moco_aux_coord_tracking_weight=_pick('moco_aux_coord_tracking_weight', 'moco_aux_coord_tracking_weight'), moco_reference_lowpass_hz=0.0 if bool(getattr(args, 'moco_no_reference_lowpass', False)) else float(getattr(args, 'moco_reference_lowpass_hz', None) if getattr(args, 'moco_reference_lowpass_hz', None) is not None else base.moco_reference_lowpass_hz), moco_apply_tracked_states_to_guess=not bool(getattr(args, 'moco_no_apply_tracked_guess', False)), moco_minimize_implicit_aux_derivatives=not bool(getattr(args, 'moco_no_implicit_aux_derivatives', False)), moco_weld_toe_joints=not bool(getattr(args, 'moco_no_weld_toes', False)), moco_multi_contact=not bool(getattr(args, 'moco_no_multi_contact', False)), moco_adaptive_mesh=not bool(getattr(args, 'moco_no_adaptive_mesh', False)), moco_adaptive_mesh_speed_deg_s=_pick('moco_adaptive_mesh_speed_deg_s', 'moco_adaptive_mesh_speed_deg_s'), moco_adaptive_mesh_interval=_pick('moco_adaptive_mesh_interval', 'moco_adaptive_mesh_interval'), moco_contact_toe_radius_m=_pick('moco_contact_toe_radius_m', 'moco_contact_toe_radius_m'), moco_use_function_based_paths=not bool(getattr(args, 'moco_no_function_based_paths', False)), moco_parallel_segments=int(parallel_seg) if parallel_seg is not None else base.moco_parallel_segments, opensim_log_level=str(getattr(args, 'opensim_log_level', base.opensim_log_level)), keep_temp=bool(keep_temp), moco_core_duration_s=float(getattr(args, 'moco_core_duration_s', None) if getattr(args, 'moco_core_duration_s', None) is not None else base.moco_core_duration_s), moco_buffer_duration_s=float(getattr(args, 'moco_buffer_duration_s', None) if getattr(args, 'moco_buffer_duration_s', None) is not None else base.moco_buffer_duration_s), moco_stitch_blend_s=float(getattr(args, 'moco_stitch_blend_s', None) if getattr(args, 'moco_stitch_blend_s', None) is not None else base.moco_stitch_blend_s))

def _normalize_opensim_log_level(level: str) -> str:
    key = str(level).strip().lower()
    if key in ('', 'silent', 'none', 'quiet'):
        return 'Off'
    return str(level).strip()

def configure_opensim_logging(level: str='Off') -> None:
    try:
        osim.Logger.setLevelString(_normalize_opensim_log_level(level))
    except Exception:
        pass

def _opensim_stdio_suppressed(level: str) -> bool:
    return _normalize_opensim_log_level(level) == 'Off'

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
def opensim_quiet(level: str='Off'):
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

def muscle_names(model: Any | None=None) -> Tuple[str, ...]:
    if model is None:
        with opensim_quiet('Off'):
            model = osim.Model(str(rajagopal_model_path()))
            model.initSystem()
    muscles = model.getMuscles()
    return tuple((muscles.get(i).getName() for i in range(muscles.getSize())))

def activation_column_for_muscle(label: str, muscle: str) -> bool:
    lab = str(label).strip()
    if lab == muscle:
        return True
    needle = f'/{muscle}/activation'
    return lab.endswith(needle) or needle in lab

def _storage_to_array(storage: Any) -> Tuple[np.ndarray, List[str]]:
    labels: List[str] = []
    for i in range(storage.getColumnLabels().size()):
        labels.append(storage.getColumnLabels().get(i))
    rows: List[List[float]] = []
    times: List[float] = []
    for i in range(storage.getSize()):
        sv = storage.getStateVector(i)
        d = sv.getData()
        times.append(float(sv.getTime()))
        rows.append([float(d.get(j)) for j in range(d.size())])
    data = np.asarray(rows, dtype=np.float64)
    if not labels:
        return (data, labels)
    if str(labels[0]).strip().lower() == 'time' and data.ndim == 2:
        if data.shape[1] == len(labels) - 1:
            data = np.column_stack([np.asarray(times, dtype=np.float64), data])
        elif data.shape[1] == len(labels):
            data[:, 0] = np.asarray(times, dtype=np.float64)
    return (data, labels)

def _validate_q_input(q: np.ndarray) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f'Expected q [T, ndof], got {arr.shape}')
    if arr.shape[1] != len(RAJAGOPAL_NIMBLE_DOF_NAMES):
        raise ValueError(f'Expected q [T, {len(RAJAGOPAL_NIMBLE_DOF_NAMES)}], got {arr.shape}')
    return arr

def _activation_work_dir(cfg: MuscleActivationConfig, *, prefix: str) -> Tuple[Path, bool]:
    if cfg.temp_dir:
        work_dir = Path(cfg.temp_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        return (work_dir, False)
    work_dir = Path(tempfile.mkdtemp(prefix=prefix))
    return (work_dir, not bool(cfg.keep_temp))

def compute_muscle_activation(q: np.ndarray, *, cfg: MuscleActivationConfig | None=None) -> MuscleActivationResult:
    cfg = cfg or MuscleActivationConfig()
    method = normalize_activation_method(cfg.activation_method)
    configure_opensim_logging(cfg.opensim_log_level)
    arr = _validate_q_input(q)
    prefix = 'sindyffuse_opensimad_' if method == 'opensimad' else 'sindyffuse_moco_'
    work_dir, cleanup = _activation_work_dir(cfg, prefix=prefix)
    try:
        if method == 'opensimad':
            from nimble.opensimad_track import run_opensimad_track
            return run_opensimad_track(arr, cfg=cfg, work_dir=work_dir)
        from nimble.moco_track import run_moco_track
        return run_moco_track(arr, cfg=cfg, work_dir=work_dir)
    finally:
        if cleanup:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)

def activation_stats(activations: np.ndarray) -> Dict[str, np.ndarray]:
    act = np.asarray(activations, dtype=np.float32)
    if act.ndim != 2:
        raise ValueError(f'Expected activations [T, M], got {act.shape}')
    mean_activation = np.nanmean(act, axis=1)
    max_activation = np.nanmax(act, axis=1)
    if act.shape[0] > 1:
        diff = np.diff(act, axis=0)
        smooth_tail = np.linalg.norm(diff, axis=1)
        activation_smoothness = np.concatenate([np.asarray([0.0], dtype=np.float32), smooth_tail.astype(np.float32)])
    else:
        activation_smoothness = np.zeros((1,), dtype=np.float32)
    return {'mean_activation': mean_activation.astype(np.float32), 'max_activation': max_activation.astype(np.float32), 'activation_smoothness': activation_smoothness.astype(np.float32)}