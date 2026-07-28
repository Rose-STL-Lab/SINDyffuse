"""Thin wrapper over musint for HumanML3D ↔ MinT muscle label lookup."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from common.paths import default_mint_root
from common.skeleton_config import DEFAULT_FPS
from osim.muscle_schema import MINT_MUSCLE_COUNT, mint_muscle_names


@dataclass(frozen=True)
class MintLabelResult:
    activations: np.ndarray  # [T, 402]
    start_frame: int
    end_frame: int
    has_labels: bool
    fps: float = DEFAULT_FPS


@lru_cache(maxsize=1)
def _mint_dataset(mint_root: str):
    from musint.datasets.mint_dataset import MintDataset

    return MintDataset(str(mint_root), use_cache=True, load_humanml3d_names=True)


def lookup_hml_motion(
    motion_id: str,
    *,
    mint_root: Optional[str] = None,
    num_frames: Optional[int] = None,
    target_fps: float = DEFAULT_FPS,
    rolling_average: bool = False,
) -> MintLabelResult:
    """Load MinT muscle activations aligned to a HumanML3D motion id."""
    root = str(mint_root or default_mint_root())
    ds = _mint_dataset(root)
    try:
        mint_data, frames = ds.by_humanml3d_name(str(motion_id), as_time=False)
    except ValueError:
        t = int(num_frames or 0)
        empty = np.zeros((max(t, 0), MINT_MUSCLE_COUNT), dtype=np.float32)
        return MintLabelResult(
            activations=empty,
            start_frame=0,
            end_frame=max(t, 0),
            has_labels=False,
        )

    start_f, end_f = int(frames[0]), int(frames[1])
    if num_frames is not None:
        end_f = min(end_f, start_f + int(num_frames))

    time_window = (
        start_f / float(target_fps),
        end_f / float(target_fps),
    )
    target_count = max(0, end_f - start_f)
    try:
        from musint.utils.dataframe_utils import trim_mint_dataframe

        arr = trim_mint_dataframe(
            mint_data.muscle_activations,
            time_window,
            target_fps=float(target_fps),
            rolling_average=bool(rolling_average),
            target_frame_count=target_count if target_count > 0 else None,
            as_numpy=True,
        )
        act = np.asarray(arr, dtype=np.float32)
        if act.ndim == 1:
            act = act.reshape(-1, 1)
        if act.shape[1] != MINT_MUSCLE_COUNT:
            raise ValueError(f"Expected {MINT_MUSCLE_COUNT} muscles, got {act.shape[1]}")
        # Reorder columns to canonical MUSINT_402 if dataframe column order differs
        act = _reorder_to_canonical(mint_data, act)
        gaps = mint_data.get_gaps(as_frame=True, target_fps=float(target_fps))
        act = _mask_gap_frames(act, gaps, start_frame=start_f, end_frame=end_f)
        return MintLabelResult(
            activations=act,
            start_frame=start_f,
            end_frame=end_f,
            has_labels=True,
        )
    except Exception:
        t = target_count
        empty = np.zeros((t, MINT_MUSCLE_COUNT), dtype=np.float32)
        return MintLabelResult(
            activations=empty,
            start_frame=start_f,
            end_frame=end_f,
            has_labels=False,
        )


def _reorder_to_canonical(mint_data, act: np.ndarray) -> np.ndarray:
    """Ensure muscle columns match ``mint_muscle_names()`` ordering."""
    canonical = mint_muscle_names()
    df_cols = list(mint_data.muscle_activations.columns)
    if tuple(df_cols) == canonical:
        return act
    if len(df_cols) != len(canonical):
        return act
    index = {name: i for i, name in enumerate(df_cols)}
    order = [index[name] for name in canonical if name in index]
    if len(order) != len(canonical):
        return act
    return act[:, order].astype(np.float32)


def _mask_gap_frames(
    act: np.ndarray,
    gaps: list,
    *,
    start_frame: int,
    end_frame: int,
) -> np.ndarray:
    """Zero activations inside MinT simulation gaps (relative to segment)."""
    if not gaps or act.size == 0:
        return act
    out = act.copy()
    for g0, g1 in gaps:
        if g1 <= start_frame or g0 >= end_frame:
            continue
        lo = max(0, int(g0) - int(start_frame))
        hi = min(out.shape[0], int(g1) - int(start_frame))
        if hi > lo:
            out[lo:hi] = 0.0
    return out


def motion_has_mint_labels(motion_id: str, *, mint_root: Optional[str] = None) -> bool:
    root = str(mint_root or default_mint_root())
    try:
        ds = _mint_dataset(root)
        ds.by_humanml3d_name(str(motion_id))
        return True
    except Exception:
        return False
