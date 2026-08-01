from __future__ import annotations
import warnings
from typing import Any, List, Tuple
import numpy as np
from nimble.b3d_schema import GUIDANCE_FEATURES, MUSCLE_ACTIVATIONS, MUSCLE_ACTIVATION_MASK, SIM_GRF, SINDY_FEATURES, unpack_activation_mask, unpack_guidance_features, unpack_muscle_activations, unpack_sim_grf, unpack_sindy_features
from nimble.muscle_b3d import MUSCLE_ACTIVATION_ROWS
_WARNED_MISSING: set[str] = set()

def subject_has_custom_value(subj: Any, name: str) -> bool:
    try:
        return str(name) in list(subj.getCustomValues())
    except Exception:
        return False

def _frame_custom_vector(frame: Any, name: str) -> np.ndarray:
    for key, vec in frame.customValues:
        if str(key) == str(name):
            return np.asarray(vec, dtype=np.float64).reshape(-1)
    raise KeyError(f"customValues missing {name!r} on frame {getattr(frame, 't', '?')}")

def _frame_has_custom(frame: Any, name: str) -> bool:
    return any((str(key) == str(name) for key, _ in frame.customValues))

def read_custom_frames(subj: Any, trial: int, start_frame: int, num_frames: int, name: str) -> np.ndarray:
    frames = subj.readFrames(trial=int(trial), startFrame=int(start_frame), numFramesToRead=int(num_frames), includeSensorData=True, includeProcessingPasses=False)
    if not frames:
        raise RuntimeError(f'Empty B3D read trial={trial} start={start_frame} n={num_frames} name={name!r}')
    rows: List[np.ndarray] = []
    for fr in frames:
        rows.append(_frame_custom_vector(fr, name))
    return np.stack(rows, axis=0).astype(np.float32, copy=False)

def read_guidance_features_frames(subj: Any, trial: int, start_frame: int, num_frames: int) -> np.ndarray:
    raw = read_custom_frames(subj, trial, start_frame, num_frames, GUIDANCE_FEATURES)
    return unpack_guidance_features(raw)

def read_muscle_activations_frames(subj: Any, trial: int, start_frame: int, num_frames: int) -> np.ndarray:
    raw = read_custom_frames(subj, trial, start_frame, num_frames, MUSCLE_ACTIVATIONS)
    return unpack_muscle_activations(raw)

def read_sindy_features_frames(subj: Any, trial: int, start_frame: int, num_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    raw = read_custom_frames(subj, trial, start_frame, num_frames, SINDY_FEATURES)
    return unpack_sindy_features(raw)

def read_sim_grf_frames(subj: Any, trial: int, start_frame: int, num_frames: int) -> np.ndarray:
    raw = read_custom_frames(subj, trial, start_frame, num_frames, SIM_GRF)
    return unpack_sim_grf(raw)

def read_muscle_activation_mask_frames(subj: Any, trial: int, start_frame: int, num_frames: int) -> np.ndarray:
    raw = read_custom_frames(subj, trial, start_frame, num_frames, MUSCLE_ACTIVATION_MASK)
    return unpack_activation_mask(raw)

def warn_missing_custom_once(b3d_path: str, name: str) -> None:
    key = f'{b3d_path}:{name}'
    if key in _WARNED_MISSING:
        return
    _WARNED_MISSING.add(key)
    warnings.warn(f'B3D {b3d_path} missing customValues {name!r}; computing at load time. Re-run preprocess_nimble.py to embed cached features.', stacklevel=3)

def b3d_has_cached_sindy_features(subj: Any, trial: int=0) -> bool:
    if not (subject_has_custom_value(subj, GUIDANCE_FEATURES) and subject_has_custom_value(subj, SINDY_FEATURES)):
        return False
    try:
        frames = subj.readFrames(trial=int(trial), startFrame=0, numFramesToRead=1, includeSensorData=True, includeProcessingPasses=False)
    except Exception:
        return False
    if not frames:
        return False
    fr = frames[0]
    return _frame_has_custom(fr, GUIDANCE_FEATURES) and _frame_has_custom(fr, SINDY_FEATURES)

def b3d_has_muscle_activations(subj: Any, trial: int=0) -> bool:
    if not subject_has_custom_value(subj, MUSCLE_ACTIVATIONS):
        return False
    try:
        dim = int(subj.getCustomValueDim(MUSCLE_ACTIVATIONS))
    except Exception:
        dim = 0
    if dim != int(MUSCLE_ACTIVATION_ROWS):
        return False
    try:
        frames = subj.readFrames(trial=int(trial), startFrame=0, numFramesToRead=1, includeSensorData=True, includeProcessingPasses=False)
    except Exception:
        return False
    if not frames:
        return False
    return _frame_has_custom(frames[0], MUSCLE_ACTIVATIONS)

def b3d_has_sindyffuse_custom_values(subj: Any, trial: int=0) -> bool:
    return b3d_has_muscle_activations(subj, trial=trial) and b3d_has_cached_sindy_features(subj, trial=trial)