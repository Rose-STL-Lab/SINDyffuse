"""High-level loaders: HumanML3D ``.npy`` → in-memory feature trials."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from osim.features import compute_features_from_hml3d
from osim.moco_runtime import moco_marker_track_feature_summary
from common.paths import DEFAULT_MODEL_PATH


def build_memory_result(
    arr: np.ndarray,
    trial_name: str,
    include_poses: bool,
    fps: int,
    sampling_frequency: float | None,
    model_path: str | None,
    *,
    smooth_poses: bool = False,
    smooth_cutoff_hz: float = 6.0,
    smooth_butterworth_order: int = 2,
) -> Dict[str, Any]:
    dt = 1.0 / float(sampling_frequency if sampling_frequency is not None else fps)
    feats, smooth_meta = compute_features_from_hml3d(
        arr.astype(np.float32),
        dt=dt,
        include_poses=include_poses,
        model_path=model_path,
        fps=int(fps),
        sampling_frequency=sampling_frequency,
        smooth_poses=smooth_poses,
        smooth_cutoff_hz=smooth_cutoff_hz,
        smooth_butterworth_order=smooth_butterworth_order,
    )
    num_dofs = int(feats["velocities"].shape[1])
    num_frames = int(feats["velocities"].shape[0])

    mp = model_path if model_path else DEFAULT_MODEL_PATH
    summ_m = moco_marker_track_feature_summary(
        arr.astype(np.float64),
        model_path=str(mp),
        dt=float(dt),
        fps=float(sampling_frequency if sampling_frequency is not None else fps),
        max_frames=num_frames,
        smooth_before_track=bool(smooth_poses),
        smooth_cutoff_hz=float(smooth_cutoff_hz),
        smooth_butterworth_order=int(smooth_butterworth_order),
    )
    for key, val in summ_m.items():
        feats[key] = val

    return {
        "source": trial_name,
        "pose_smoothing": smooth_meta,
        "trials": [
            {
                "trial_idx": 0,
                "trial_name": trial_name,
                "num_frames": num_frames,
                "num_dofs": num_dofs,
                "dt": float(dt),
                "features": feats,
            }
        ],
    }


def from_poses(
    poses: np.ndarray,
    trial_name: str = "in_memory_trial",
    processing_pass: int = 0,
    include_sensor_data: bool = True,
    include_poses: bool = False,
    fps: int = 20,
    sampling_frequency: float | None = None,
    model_path: str | None = DEFAULT_MODEL_PATH,
    *,
    smooth_poses: bool = False,
    smooth_cutoff_hz: float = 6.0,
    smooth_butterworth_order: int = 2,
) -> Dict[str, Any]:
    _ = processing_pass, include_sensor_data  # kept for API compatibility
    return build_memory_result(
        arr=poses,
        trial_name=str(trial_name),
        include_poses=bool(include_poses),
        fps=int(fps),
        sampling_frequency=sampling_frequency,
        model_path=model_path,
        smooth_poses=smooth_poses,
        smooth_cutoff_hz=smooth_cutoff_hz,
        smooth_butterworth_order=smooth_butterworth_order,
    )


def from_hml3d(
    input_path: str,
    processing_pass: int = 0,
    include_sensor_data: bool = True,
    include_poses: bool = False,
    fps: int = 20,
    sampling_frequency: float | None = None,
    model_path: str | None = DEFAULT_MODEL_PATH,
    *,
    smooth_poses: bool = False,
    smooth_cutoff_hz: float = 6.0,
    smooth_butterworth_order: int = 2,
) -> Dict[str, Any]:
    arr = np.load(input_path).astype(np.float32)
    stem = Path(input_path).stem
    memory_result = from_poses(
        poses=arr,
        trial_name=stem,
        processing_pass=processing_pass,
        include_sensor_data=include_sensor_data,
        include_poses=include_poses,
        fps=fps,
        sampling_frequency=sampling_frequency,
        model_path=model_path,
        smooth_poses=smooth_poses,
        smooth_cutoff_hz=smooth_cutoff_hz,
        smooth_butterworth_order=smooth_butterworth_order,
    )
    return {"input_npy": str(input_path), "memory_result": memory_result, "convert_stats": {"engine": "opensim_moco"}}


def from_hml3d_dir(
    input_dir: str,
    processing_pass: int = 0,
    include_sensor_data: bool = True,
    include_poses: bool = False,
    fps: int = 20,
    sampling_frequency: float | None = None,
    model_path: str | None = DEFAULT_MODEL_PATH,
    *,
    smooth_poses: bool = False,
    smooth_cutoff_hz: float = 6.0,
    smooth_butterworth_order: int = 2,
) -> Dict[str, Any]:
    npy_files = sorted(Path(input_dir).rglob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found under {input_dir}")
    rows: List[Dict[str, Any]] = []
    for npy in npy_files:
        rows.append(
            from_hml3d(
                input_path=str(npy),
                processing_pass=processing_pass,
                include_sensor_data=include_sensor_data,
                include_poses=include_poses,
                fps=fps,
                sampling_frequency=sampling_frequency,
                model_path=model_path,
                smooth_poses=smooth_poses,
                smooth_cutoff_hz=smooth_cutoff_hz,
                smooth_butterworth_order=smooth_butterworth_order,
            )
        )
    return {"input_dir": str(input_dir), "num_files": len(npy_files), "rows": rows}


def from_array(
    poses: np.ndarray,
    trial_name: str = "in_memory_trial",
    processing_pass: int = 0,
    include_sensor_data: bool = True,
    include_poses: bool = False,
    fps: int = 20,
    sampling_frequency: float | None = None,
    model_path: str | None = DEFAULT_MODEL_PATH,
    *,
    smooth_poses: bool = False,
    smooth_cutoff_hz: float = 6.0,
    smooth_butterworth_order: int = 2,
) -> Dict[str, Any]:
    return from_poses(
        poses=poses,
        trial_name=trial_name,
        processing_pass=processing_pass,
        include_sensor_data=include_sensor_data,
        include_poses=include_poses,
        fps=fps,
        sampling_frequency=sampling_frequency,
        model_path=model_path,
        smooth_poses=smooth_poses,
        smooth_cutoff_hz=smooth_cutoff_hz,
        smooth_butterworth_order=smooth_butterworth_order,
    )
