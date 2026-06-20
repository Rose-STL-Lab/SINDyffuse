"""Read SINDyffuse B3D customValues (guidance_features, sindy_features)."""

from __future__ import annotations

import warnings
from typing import Any, List, Tuple

import numpy as np

from nimble.b3d_schema import (
    GUIDANCE_FEATURES,
    MUSCLE_ACTIVATIONS,
    SINDY_FEATURES,
    unpack_guidance_features,
    unpack_muscle_activations,
    unpack_sindy_features,
)

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
    return any(str(key) == str(name) for key, _ in frame.customValues)


def read_custom_frames(
    subj: Any,
    trial: int,
    start_frame: int,
    num_frames: int,
    name: str,
) -> np.ndarray:
    """Read one custom channel across frames → ``[num_frames, rows]`` float32."""
    # Trial-level customValues from preprocess_nimble are only exposed on frames when
    # includeSensorData=True (see nimblephysics SubjectOnDisk.readFrames).
    frames = subj.readFrames(
        trial=int(trial),
        startFrame=int(start_frame),
        numFramesToRead=int(num_frames),
        includeSensorData=True,
        includeProcessingPasses=False,
    )
    if not frames:
        raise RuntimeError(
            f"Empty B3D read trial={trial} start={start_frame} n={num_frames} name={name!r}"
        )
    rows: List[np.ndarray] = []
    for fr in frames:
        rows.append(_frame_custom_vector(fr, name))
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def read_guidance_features_frames(
    subj: Any,
    trial: int,
    start_frame: int,
    num_frames: int,
) -> np.ndarray:
    """``[num_frames, C]`` guidance channel scalars."""
    raw = read_custom_frames(subj, trial, start_frame, num_frames, GUIDANCE_FEATURES)
    return unpack_guidance_features(raw)


def read_muscle_activations_frames(
    subj: Any,
    trial: int,
    start_frame: int,
    num_frames: int,
) -> np.ndarray:
    """``[num_frames, M]`` muscle activation scalars."""
    raw = read_custom_frames(subj, trial, start_frame, num_frames, MUSCLE_ACTIVATIONS)
    return unpack_muscle_activations(raw)


def read_sindy_features_frames(
    subj: Any,
    trial: int,
    start_frame: int,
    num_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """``u`` ``[T, U]``, ``c`` ``[T, C]`` from stored sindy_features."""
    raw = read_custom_frames(subj, trial, start_frame, num_frames, SINDY_FEATURES)
    return unpack_sindy_features(raw)


def warn_missing_custom_once(b3d_path: str, name: str) -> None:
    key = f"{b3d_path}:{name}"
    if key in _WARNED_MISSING:
        return
    _WARNED_MISSING.add(key)
    warnings.warn(
        f"B3D {b3d_path} missing customValues {name!r}; computing at load time. "
        f"Re-run preprocess_nimble.py to embed cached features.",
        stacklevel=3,
    )


def b3d_has_sindyffuse_custom_values(subj: Any, trial: int = 0) -> bool:
    """True when guidance/sindy custom channels are readable on disk (not just named)."""
    if not (
        subject_has_custom_value(subj, GUIDANCE_FEATURES)
        and subject_has_custom_value(subj, SINDY_FEATURES)
        and subject_has_custom_value(subj, MUSCLE_ACTIVATIONS)
    ):
        return False
    try:
        frames = subj.readFrames(
            trial=int(trial),
            startFrame=0,
            numFramesToRead=1,
            includeSensorData=True,
            includeProcessingPasses=False,
        )
    except Exception:
        return False
    if not frames:
        return False
    fr = frames[0]
    return (
        _frame_has_custom(fr, GUIDANCE_FEATURES)
        and _frame_has_custom(fr, SINDY_FEATURES)
        and _frame_has_custom(fr, MUSCLE_ACTIVATIONS)
    )
