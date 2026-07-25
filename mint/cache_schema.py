"""NPZ cache schema for MinT q + muscle activations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from common.skeleton_config import DEFAULT_FPS, MINT_MUSCLE_COUNT
from mint.muscle_schema import mint_muscle_names

KEY_Q = "q"
KEY_MUSCLE = "muscle_activations"
KEY_GUIDANCE = "guidance_features"
KEY_SINDY = "sindy_features"
KEY_HAS_LABELS = "has_mint_labels"
KEY_FPS = "fps"

REQUIRED_KEYS = (KEY_Q, KEY_MUSCLE)


def validate_shapes(
    q: np.ndarray,
    muscle: np.ndarray,
    *,
    expected_muscles: int = MINT_MUSCLE_COUNT,
) -> None:
    if q.ndim != 2:
        raise ValueError(f"Expected q [T, ndof], got {q.shape}")
    if muscle.ndim != 2:
        raise ValueError(f"Expected muscle_activations [T, M], got {muscle.shape}")
    if q.shape[0] != muscle.shape[0]:
        raise ValueError(f"Length mismatch q T={q.shape[0]} vs muscle T={muscle.shape[0]}")
    if muscle.shape[1] != int(expected_muscles):
        raise ValueError(f"Expected {expected_muscles} muscles, got {muscle.shape[1]}")


def write_motion_cache(
    path: str | Path,
    *,
    q: np.ndarray,
    muscle_activations: np.ndarray,
    guidance_features: Optional[np.ndarray] = None,
    sindy_features: Optional[np.ndarray] = None,
    has_mint_labels: bool = True,
    fps: float = DEFAULT_FPS,
) -> None:
    q_arr = np.asarray(q, dtype=np.float32)
    act_arr = np.asarray(muscle_activations, dtype=np.float32)
    validate_shapes(q_arr, act_arr)
    payload: Dict[str, Any] = {
        KEY_Q: q_arr,
        KEY_MUSCLE: act_arr,
        KEY_HAS_LABELS: np.bool_(has_mint_labels),
        KEY_FPS: np.float32(fps),
    }
    if guidance_features is not None:
        g = np.asarray(guidance_features, dtype=np.float32)
        if g.shape[0] != q_arr.shape[0]:
            raise ValueError("guidance_features length mismatch")
        payload[KEY_GUIDANCE] = g
    if sindy_features is not None:
        s = np.asarray(sindy_features, dtype=np.float32)
        if s.shape[0] != q_arr.shape[0]:
            raise ValueError("sindy_features length mismatch")
        payload[KEY_SINDY] = s
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)


def read_motion_cache(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=False)
    out = {k: data[k] for k in data.files}
    return out


def read_q_segment(
    path: str | Path,
    start: int,
    num_frames: int,
) -> np.ndarray:
    data = read_motion_cache(path)
    q = data[KEY_Q]
    st = max(0, int(start))
    ed = min(int(q.shape[0]), st + int(num_frames))
    if ed <= st:
        raise RuntimeError(f"Empty q segment in {path} start={start} n={num_frames}")
    return q[st:ed].astype(np.float32)


def read_muscle_segment(
    path: str | Path,
    start: int,
    num_frames: int,
) -> np.ndarray:
    data = read_motion_cache(path)
    act = data[KEY_MUSCLE]
    st = max(0, int(start))
    ed = min(int(act.shape[0]), st + int(num_frames))
    if ed <= st:
        raise RuntimeError(f"Empty muscle segment in {path}")
    return act[st:ed].astype(np.float32)


def cache_has_labels(path: str | Path) -> bool:
    data = read_motion_cache(path)
    if KEY_HAS_LABELS in data:
        return bool(np.asarray(data[KEY_HAS_LABELS]).item())
    act = data[KEY_MUSCLE]
    from mint.muscle_schema import validate_activation_matrix

    return not validate_activation_matrix(act)


def write_cache_metadata(cache_dir: str | Path, *, ndof: int, fps: float = DEFAULT_FPS) -> Path:
    root = Path(cache_dir).expanduser().resolve()
    meta = {
        "skeleton": "mint",
        "feature_type": "mint_q",
        "num_dofs": int(ndof),
        "num_muscles": int(MINT_MUSCLE_COUNT),
        "muscle_names": list(mint_muscle_names()),
        "fps": float(fps),
    }
    path = root / "metadata.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_cache_metadata(cache_dir: str | Path) -> Dict[str, Any]:
    path = Path(cache_dir).expanduser().resolve() / "metadata.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
