"""OpenSim static optimization muscle activations from Nimble Rajagopal ``q``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import opensim as osim

from nimble.muscle_activation import (
    MuscleActivationConfig,
    MuscleActivationResult,
    apply_activation_postprocess,
    _storage_to_array,
    opensim_quiet,
)
from nimble.rajagopal_coord_map import build_rajagopal_coord_mapping, write_coordinates_mot
from nimble.rajagopal_model import prepare_rajagopal_activation_model

_STATIC_TOOL_NAME = "sindyffuse_static"


def _is_reserve_column(label: str) -> bool:
    lower = str(label).lower()
    return ("reserve" in lower or "residual" in lower) and "activation" not in lower


def _parse_activation_sto(
    sto_path: Path,
    muscle_name_list: Sequence[str],
    frame_times: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Parse static-opt activation ``.sto`` onto uniform ``frame_times``."""
    storage = osim.Storage(str(sto_path))
    data, labels = _storage_to_array(storage)
    del storage
    if not labels or data.size == 0:
        raise RuntimeError(f"Empty static optimization activation storage: {sto_path}")

    col_offset = 1 if labels and str(labels[0]).strip().lower() == "time" else 0
    if col_offset:
        times = np.asarray(data[:, 0], dtype=np.float64)
    else:
        times = np.arange(int(data.shape[0]), dtype=np.float64)
    label_to_col: Dict[str, int] = {}
    for i, lab in enumerate(labels):
        if str(lab).strip().lower() == "time":
            continue
        label_to_col[str(lab)] = int(i - col_offset)

    n_frames = int(frame_times.shape[0])
    n_muscles = len(muscle_name_list)
    activations = np.full((n_frames, n_muscles), np.nan, dtype=np.float64)

    for mi, name in enumerate(muscle_name_list):
        col_idx = label_to_col.get(name)
        if col_idx is None:
            continue
        col = data[:, col_idx]
        finite = np.isfinite(times) & np.isfinite(col)
        if finite.sum() < 1:
            continue
        if finite.sum() == 1:
            activations[:, mi] = float(col[finite][0])
        else:
            activations[:, mi] = np.interp(
                frame_times,
                times[finite],
                col[finite],
                left=np.nan,
                right=np.nan,
            )

    reserve_peaks: Dict[str, float] = {}
    reserve_series: List[np.ndarray] = []
    for lab, idx in label_to_col.items():
        if lab == "time" or not _is_reserve_column(lab):
            continue
        col = np.abs(np.asarray(data[:, idx], dtype=np.float64))
        reserve_peaks[str(lab)] = float(np.nanmax(col)) if col.size else 0.0
        finite = np.isfinite(times) & np.isfinite(col)
        if finite.sum() >= 1:
            reserve_series.append(
                np.interp(frame_times, times[finite], col[finite], left=0.0, right=0.0)
            )

    if reserve_series:
        reserve_max = np.max(np.stack(reserve_series, axis=0), axis=0)
    else:
        reserve_max = np.zeros(n_frames, dtype=np.float64)

    meta: Dict[str, Any] = {
        "static_reserve_peaks": reserve_peaks,
        "static_max_reserve_per_frame": reserve_max.astype(np.float64),
    }
    return activations.astype(np.float32), meta


def run_static_optimization(
    q: np.ndarray,
    *,
    cfg: MuscleActivationConfig,
    work_dir: Path,
) -> MuscleActivationResult:
    """Per-frame static optimization on prescribed Rajagopal kinematics."""
    arr = np.asarray(q, dtype=np.float64)
    t_len = int(arr.shape[0])
    if t_len < 2:
        raise ValueError(f"Need at least 2 frames for static optimization, got {t_len}")

    fps = float(cfg.fps)
    dt = 1.0 / max(fps, 1e-8)
    frame_times = np.arange(t_len, dtype=np.float64) * dt
    t0, t1 = 0.0, max(0.0, (t_len - 1) * dt - 1e-6)

    model_path, muscle_name_list = prepare_rajagopal_activation_model(
        work_dir, cfg, weld_toe_joints=False
    )
    mapping = build_rajagopal_coord_mapping(model_path=model_path)
    mot_path = write_coordinates_mot(arr, work_dir / "coordinates.mot", fps=fps, mapping=mapping)

    results_dir = work_dir / "static_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with opensim_quiet(cfg.opensim_log_level):
        tool = osim.AnalyzeTool()
        tool.setName(_STATIC_TOOL_NAME)
        tool.setModelFilename(str(model_path))
        tool.setCoordinatesFileName(str(mot_path))
        tool.setInitialTime(float(t0))
        tool.setFinalTime(float(t1))
        tool.setResultsDir(str(results_dir))
        tool.setLowpassCutoffFrequency(float(cfg.static_lowpass_cutoff_hz))

        so = osim.StaticOptimization()
        so.setUseModelForceSet(True)
        so.setActivationExponent(float(cfg.static_activation_exponent))
        so.setUseMusclePhysiology(True)
        so.setConvergenceCriterion(float(cfg.static_convergence_criterion))
        so.setMaxIterations(int(cfg.static_max_iterations))
        tool.getAnalysisSet().cloneAndAppend(so)

        setup_path = work_dir / "static_analyze_setup.xml"
        tool.printToXML(str(setup_path))
        runner = osim.AnalyzeTool(str(setup_path))
        if not runner.run():
            raise RuntimeError("OpenSim StaticOptimization AnalyzeTool.run() returned false")

        act_sto = results_dir / f"{_STATIC_TOOL_NAME}_StaticOptimization_activation.sto"
        if not act_sto.is_file():
            analysis = osim.StaticOptimization.safeDownCast(
                runner.getAnalysisSet().get(0)
            )
            storage = analysis.getActivationStorage()
            if storage is None or storage.getSize() < 1:
                raise RuntimeError(
                    "Static optimization produced no activation storage "
                    f"(work_dir={work_dir})"
                )
            analysis.printResults(_STATIC_TOOL_NAME, str(results_dir))

        if not act_sto.is_file():
            raise RuntimeError(f"Missing static optimization output: {act_sto}")

        activations, parse_meta = _parse_activation_sto(
            act_sto, muscle_name_list, frame_times
        )

    repair_meta: Dict[str, Any] = {}
    if cfg.interpolate_activations or float(cfg.activation_smooth_hz) > 0.0:
        activations, repair_meta = apply_activation_postprocess(activations, cfg)
    repaired_count = int(repair_meta.get("repaired_frame_count", 0))

    meta: Dict[str, Any] = {
        "activation_method": "static_optimization",
        "repaired_frame_count": repaired_count,
        "interpolated_frame_count": int(repair_meta.get("interpolated_frame_count", 0)),
        "extrapolated_frame_count": int(repair_meta.get("extrapolated_frame_count", 0)),
        "static_activation_exponent": float(cfg.static_activation_exponent),
        "static_lowpass_cutoff_hz": float(cfg.static_lowpass_cutoff_hz),
        "moco_max_reserve_fraction": float(cfg.moco_max_reserve_fraction),
        **{k: v for k, v in parse_meta.items() if k != "static_max_reserve_per_frame"},
    }
    meta.update(repair_meta)
    return MuscleActivationResult(
        activations=activations,
        muscle_names=tuple(muscle_name_list),
        metadata=meta,
    )
