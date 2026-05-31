"""OpenSim Static Optimization muscle activations from Nimble Rajagopal ``q``.

Standalone module for offline activation estimation. Designed for future integration:

- **B3D preprocess**: cache ``muscle_activations`` as ``[M, T]`` custom values (see ``b3d_schema``).
- **SINDy**: optional extra target block ``[T, M]`` aligned with ``targets_for_theta`` (``T-1``).
- **Diffusion guidance**: compare generated vs reference activations or penalize activation smoothness.

OpenSim Static Optimization is **not differentiable**; use cached activations during training.

Limitations:
- External loads are **heuristic** (estimated GRF from foot kinematics), not measured force plates.
- Noisy IK ``q`` can yield failed SO frames; check ``success_mask``.
- Activations can be noisy without low-pass filtering or RRA-smoothed kinematics.
"""

from __future__ import annotations

import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import opensim as osim

from surrogate.opensim_coord_map import (
    RAJAGOPAL_NIMBLE_DOF_NAMES,
    build_rajagopal_coord_mapping,
    write_coordinates_mot,
)
from nimble.skeleton_registry import get_spec


def rajagopal_model_path() -> Path:
    """Path to the bundled Rajagopal 2015 ``.osim`` (same as nimblephysics)."""
    import nimblephysics as nimble

    return Path(nimble.__file__).parent / "models" / "rajagopal_data" / "Rajagopal2015.osim"


@dataclass
class MuscleActivationConfig:
    """Configuration for OpenSim Static Optimization."""

    fps: float = 20.0
    mass_kg: float = 70.0
    g_mps2: float = 9.81
    lowpass_cutoff_hz: float = 6.0
    activation_exponent: float = 2.0
    max_iterations: int = 100
    convergence_criterion: float = 1e-4
    use_muscle_physiology: bool = True
    use_reserve_actuators: bool = True
    reserve_coord_names: Tuple[str, ...] = (
        "pelvis_tx",
        "pelvis_ty",
        "pelvis_tz",
        "pelvis_tilt",
        "pelvis_list",
        "pelvis_rotation",
    )
    reserve_optimal_force: float = 1.0
    contact_height_thresh_m: float = 0.06
    contact_speed_thresh_mps: float = 1.2
    temp_dir: Optional[str] = None
    keep_temp: bool = False


@dataclass
class MuscleActivationResult:
    """Static optimization output for one motion."""

    activations: np.ndarray
    muscle_names: Tuple[str, ...]
    success_mask: np.ndarray
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
        model = osim.Model(str(rajagopal_model_path()))
        model.initSystem()
    muscles = model.getMuscles()
    return tuple(muscles.get(i).getName() for i in range(muscles.getSize()))


def _estimate_grf_from_q(
    q: np.ndarray,
    *,
    fps: float,
    mass_kg: float,
    g_mps2: float,
    height_thresh_m: float,
    speed_thresh_mps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Heuristic vertical GRF per foot from Rajagopal keypoints (numpy)."""
    from nimble.physics import load_model
    from nimble.rajagopal_kin import IDX_FOOT_L, IDX_FOOT_R, keypoints_numpy

    sk = load_model().skeleton
    kp = keypoints_numpy(sk, np.asarray(q, dtype=np.float64))
    t_len = int(kp.shape[0])
    dt = 1.0 / max(float(fps), 1e-8)
    foot_l = kp[:, IDX_FOOT_L, :]
    foot_r = kp[:, IDX_FOOT_R, :]
    com = kp.mean(axis=1)
    if t_len > 1:
        com_v = np.gradient(com, dt, axis=0)
        com_a = np.gradient(com_v, dt, axis=0)
    else:
        com_a = np.zeros_like(com)

    ground_y = float(min(foot_l[:, 1].min(), foot_r[:, 1].min()))
    left_h = foot_l[:, 1] - ground_y
    right_h = foot_r[:, 1] - ground_y
    if t_len > 1:
        left_speed = np.linalg.norm(np.diff(foot_l, axis=0, prepend=foot_l[:1]), axis=1) / dt
        right_speed = np.linalg.norm(np.diff(foot_r, axis=0, prepend=foot_r[:1]), axis=1) / dt
    else:
        left_speed = np.zeros(t_len)
        right_speed = np.zeros(t_len)

    def _contact(height: np.ndarray, speed: np.ndarray) -> np.ndarray:
        return (
            (height <= float(height_thresh_m)) & (speed <= float(speed_thresh_mps))
        ).astype(np.float64)

    c_l = _contact(left_h, left_speed)
    c_r = _contact(right_h, right_speed)
    g_vec = np.array([0.0, -float(g_mps2), 0.0], dtype=np.float64)
    f_total = float(mass_kg) * (com_a - g_vec.reshape(1, 3))
    f_y = np.clip(f_total[:, 1], 0.0, None)
    any_c = np.clip(c_l + c_r, 0.0, 1.0)
    f_y = f_y * any_c
    denom = np.clip(c_l + c_r, 1e-3, None)
    f_l = f_y * (c_l / denom)
    f_r = f_y * (c_r / denom)
    times = np.arange(t_len, dtype=np.float64) * dt
    return times, f_l, f_r


def build_external_loads_xml(
    q: np.ndarray,
    xml_path: str | Path,
    grf_mot_path: str | Path,
    *,
    cfg: MuscleActivationConfig | None = None,
    foot_bodies: Tuple[str, str] | None = None,
) -> Path:
    """Write ``ExternalLoads`` XML + GRF ``.mot`` from heuristic contact/GRF on ``q``."""
    cfg = cfg or MuscleActivationConfig()
    spec = get_spec("rajagopal")
    feet = foot_bodies or spec.foot_body_names
    xml_out = Path(xml_path)
    mot_out = Path(grf_mot_path)
    xml_out.parent.mkdir(parents=True, exist_ok=True)

    times, f_l, f_r = _estimate_grf_from_q(
        q,
        fps=float(cfg.fps),
        mass_kg=float(cfg.mass_kg),
        g_mps2=float(cfg.g_mps2),
        height_thresh_m=float(cfg.contact_height_thresh_m),
        speed_thresh_mps=float(cfg.contact_speed_thresh_mps),
    )
    t_len = int(times.shape[0])
    col_names: List[str] = []
    for foot in feet:
        col_names.extend([f"{foot}_vx", f"{foot}_vy", f"{foot}_vz"])
    with mot_out.open("w", encoding="utf-8") as f:
        f.write("ExternalLoads\n")
        f.write("version=1\n")
        f.write(f"nRows={t_len}\n")
        f.write(f"nColumns={1 + len(col_names)}\n")
        f.write("inDegrees=no\n\n")
        f.write("endheader\n")
        f.write("time\t" + "\t".join(col_names) + "\n")
        for t in range(t_len):
            row = [times[t], 0.0, f_l[t], 0.0, 0.0, f_r[t], 0.0]
            f.write("\t".join(f"{x:.8f}" for x in row) + "\n")

    xml_out.write_text(
        f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="30000">
    <ExternalLoads name="externalloads">
        <objects>
            <ExternalForce name="grf_{feet[0]}">
                <applied_to_body>{feet[0]}</applied_to_body>
                <force_expressed_in_body>ground</force_expressed_in_body>
                <point_expressed_in_body>ground</point_expressed_in_body>
                <force_identifier>{feet[0]}</force_identifier>
                <point_identifier></point_identifier>
                <torque_identifier></torque_identifier>
            </ExternalForce>
            <ExternalForce name="grf_{feet[1]}">
                <applied_to_body>{feet[1]}</applied_to_body>
                <force_expressed_in_body>ground</force_expressed_in_body>
                <point_expressed_in_body>ground</point_expressed_in_body>
                <force_identifier>{feet[1]}</force_identifier>
                <point_identifier></point_identifier>
                <torque_identifier></torque_identifier>
            </ExternalForce>
        </objects>
        <datafile>{mot_out.name}</datafile>
    </ExternalLoads>
</OpenSimDocument>
""",
        encoding="utf-8",
    )
    return xml_out


def _append_reserve_actuators(model: Any, cfg: MuscleActivationConfig) -> None:
    force_set = model.updForceSet()
    for cname in cfg.reserve_coord_names:
        coord = model.getCoordinateSet().get(str(cname))
        act = osim.CoordinateActuator()
        act.setName(f"reserve_{cname}")
        act.setCoordinate(coord)
        act.setOptimalForce(float(cfg.reserve_optimal_force))
        act.setMinControl(-1e5)
        act.setMaxControl(1e5)
        force_set.append(act)


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


def _parse_activation_storage(
    storage_path: Path,
    muscle_name_list: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    storage = osim.Storage(str(storage_path))
    data, labels = _storage_to_array(storage)
    if not labels:
        raise RuntimeError(f"Empty activation storage: {storage_path}")
    label_to_col = {lab: i for i, lab in enumerate(labels)}
    if "time" in label_to_col:
        times = data[:, label_to_col["time"]]
    else:
        times = data[:, 0]
    muscle_cols = []
    for name in muscle_name_list:
        if name not in label_to_col:
            raise KeyError(f"Muscle {name!r} missing from activation storage columns")
        muscle_cols.append(label_to_col[name])
    activations = data[:, muscle_cols].astype(np.float32)
    finite_rows = np.isfinite(activations).all(axis=1)
    return activations, finite_rows.astype(np.bool_)


def _run_static_optimization(
    q: np.ndarray,
    *,
    cfg: MuscleActivationConfig,
    work_dir: Path,
) -> MuscleActivationResult:
    t_len = int(q.shape[0])
    if t_len < 2:
        raise ValueError(f"Need at least 2 frames for Static Optimization, got {t_len}")

    model_path = rajagopal_model_path()
    mapping = build_rajagopal_coord_mapping(model_path=model_path)
    names = muscle_names()

    mot_path = work_dir / "coordinates.mot"
    write_coordinates_mot(q, mot_path, fps=float(cfg.fps), mapping=mapping)

    grf_mot = work_dir / "grf_estimates.mot"
    ext_xml = work_dir / "external_loads.xml"
    build_external_loads_xml(q, ext_xml, grf_mot, cfg=cfg)

    model = osim.Model(str(model_path))
    if cfg.use_reserve_actuators:
        _append_reserve_actuators(model, cfg)

    static_opt = osim.StaticOptimization()
    static_opt.setName("StaticOptimization")
    static_opt.setUseModelForceSet(True)
    static_opt.setActivationExponent(float(cfg.activation_exponent))
    static_opt.setUseMusclePhysiology(bool(cfg.use_muscle_physiology))
    static_opt.setConvergenceCriterion(float(cfg.convergence_criterion))
    static_opt.setMaxIterations(int(cfg.max_iterations))
    model.addAnalysis(static_opt)
    model.initSystem()

    dt = 1.0 / max(float(cfg.fps), 1e-8)
    analysis = osim.AnalyzeTool()
    analysis.setModel(model)
    analysis.setName("muscle_activation")
    analysis.setStartTime(0.0)
    analysis.setFinalTime((t_len - 1) * dt)
    analysis.setLowpassCutoffFrequency(float(cfg.lowpass_cutoff_hz))
    analysis.setCoordinatesFileName(str(mot_path))
    analysis.setExternalLoadsFileName(str(ext_xml))
    analysis.setLoadModelAndInput(True)
    analysis.setResultsDir(str(work_dir))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analysis.run()

    act_files = list(work_dir.glob("*_StaticOptimization_activation.sto"))
    if not act_files:
        raise RuntimeError(f"Static Optimization produced no activation file in {work_dir}")
    activations, success_mask = _parse_activation_storage(act_files[0], names)

    if activations.shape[0] != t_len:
        if activations.shape[0] > t_len:
            activations = activations[:t_len]
            success_mask = success_mask[:t_len]
        else:
            pad = t_len - activations.shape[0]
            activations = np.pad(activations, ((0, pad), (0, 0)), mode="constant", constant_values=np.nan)
            success_mask = np.pad(success_mask, (0, pad), constant_values=False)

    forces = None
    force_files = list(work_dir.glob("*_StaticOptimization_force.sto"))
    if force_files:
        fdata, flabels = _storage_to_array(osim.Storage(str(force_files[0])))
        muscle_force_cols = [flabels.index(n) for n in names if n in flabels]
        if muscle_force_cols:
            forces = fdata[:t_len, muscle_force_cols].astype(np.float32)

    meta: Dict[str, Any] = {
        "opensim_version": str(osim.GetVersion()),
        "model_path": str(model_path),
        "num_frames": t_len,
        "num_muscles": len(names),
        "fps": float(cfg.fps),
        "success_fraction": float(success_mask.mean()),
        "lowpass_cutoff_hz": float(cfg.lowpass_cutoff_hz),
        "used_reserve_actuators": bool(cfg.use_reserve_actuators),
        "external_loads_heuristic": True,
    }
    return MuscleActivationResult(
        activations=activations,
        muscle_names=tuple(names),
        success_mask=success_mask,
        metadata=meta,
        forces=forces,
    )


def compute_muscle_activation(
    q: np.ndarray,
    *,
    cfg: MuscleActivationConfig | None = None,
) -> MuscleActivationResult:
    """Compute full-body muscle activations from Nimble Rajagopal ``q`` ``[T, 37]``.

    Uses OpenSim Static Optimization (AnalyzeTool). Returns activations ``[T, M]``
    with ``M=80`` for Rajagopal.
    """
    cfg = cfg or MuscleActivationConfig()
    arr = np.asarray(q, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected q [T, ndof], got {arr.shape}")
    if arr.shape[1] != len(RAJAGOPAL_NIMBLE_DOF_NAMES):
        raise ValueError(
            f"Expected q [T, {len(RAJAGOPAL_NIMBLE_DOF_NAMES)}], got {arr.shape}"
        )

    if cfg.temp_dir:
        work_dir = Path(cfg.temp_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="sindyffuse_so_"))
        cleanup = not bool(cfg.keep_temp)

    try:
        result = _run_static_optimization(arr, cfg=cfg, work_dir=work_dir)
        if not np.isfinite(result.activations).all():
            raise RuntimeError(
                "Static Optimization produced non-finite muscle activations; "
                f"success_fraction={result.metadata.get('success_fraction', 0.0)}"
            )
        return result
    finally:
        if cleanup:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)


def activation_stats(activations: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-frame scalar summaries for future guidance penalties."""
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
