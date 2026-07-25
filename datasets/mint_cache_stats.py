"""Compute q-space Mean/Std over MinT NPZ cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from common.paths import mint_cache_dir
from common.skeleton_config import DEFAULT_FPS
from datasets.splits import load_split_ids
from mint.cache_schema import KEY_Q, load_cache_metadata, write_cache_metadata


def compute_mint_normalization_stats(
    data_root: str | Path,
    *,
    split: str = "train",
    splits: Sequence[str] | None = None,
    max_motions: int = 0,
    max_frames_per_motion: int = 0,
) -> dict:
    root = Path(data_root).expanduser().resolve()
    cache_dir = mint_cache_dir(root)
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Missing MinT cache: {cache_dir}")

    if splits:
        ids: list[str] = []
        for sp in splits:
            ids.extend(load_split_ids(root, str(sp)))
        ids = sorted(set(ids))
        stats_label = "+".join(str(sp) for sp in splits)
    else:
        ids = load_split_ids(root, split)
        stats_label = str(split)

    if int(max_motions) > 0:
        ids = ids[: int(max_motions)]

    sum_q = None
    sum_sq = None
    count = 0
    num_dofs = None

    for sid in ids:
        p = cache_dir / f"{sid}.npz"
        if not p.is_file():
            continue
        data = np.load(str(p))
        q = np.asarray(data[KEY_Q], dtype=np.float64)
        if int(max_frames_per_motion) > 0:
            q = q[: int(max_frames_per_motion)]
        if q.ndim != 2 or q.shape[0] < 1:
            continue
        if num_dofs is None:
            num_dofs = int(q.shape[1])
            sum_q = np.zeros(num_dofs, dtype=np.float64)
            sum_sq = np.zeros(num_dofs, dtype=np.float64)
        if int(q.shape[1]) != num_dofs:
            continue
        sum_q += q.sum(axis=0)
        sum_sq += (q * q).sum(axis=0)
        count += int(q.shape[0])

    if count < 1 or sum_q is None or sum_sq is None or num_dofs is None:
        cached = sorted(p.stem for p in cache_dir.glob("*.npz"))
        if not cached:
            raise RuntimeError(f"No frames accumulated for MinT q statistics under {root}")
        for sid in cached:
            p = cache_dir / f"{sid}.npz"
            data = np.load(str(p))
            q = np.asarray(data[KEY_Q], dtype=np.float64)
            if num_dofs is None:
                num_dofs = int(q.shape[1])
                sum_q = np.zeros(num_dofs, dtype=np.float64)
                sum_sq = np.zeros(num_dofs, dtype=np.float64)
            sum_q += q.sum(axis=0)
            sum_sq += (q * q).sum(axis=0)
            count += int(q.shape[0])
        stats_label = "cache"

    mean = (sum_q / float(count)).astype(np.float32)
    var = np.maximum(sum_sq / float(count) - mean.astype(np.float64) ** 2, 1e-12)
    std = np.sqrt(var).astype(np.float32)
    std = np.clip(std, 1e-6, None)

    np.save(cache_dir / "Mean.npy", mean)
    np.save(cache_dir / "Std.npy", std)
    write_cache_metadata(cache_dir, ndof=int(num_dofs), fps=DEFAULT_FPS)

    meta_path = cache_dir / "metadata.json"
    meta = load_cache_metadata(cache_dir)
    meta.update(
        {
            "stats_split": stats_label,
            "stats_frame_count": int(count),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "num_dofs": int(num_dofs),
        "stats_split": stats_label,
        "stats_frame_count": int(count),
        "mean_path": str(cache_dir / "Mean.npy"),
        "std_path": str(cache_dir / "Std.npy"),
    }
