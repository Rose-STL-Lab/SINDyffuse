"""Sliding windows over MinT NPZ cache for surrogate training."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from common.paths import mint_cache_dir
from common.skeleton_config import MINT_MUSCLE_COUNT
from datasets.splits import load_split_ids
from mint.cache_schema import KEY_HAS_LABELS, KEY_MUSCLE, KEY_Q, cache_has_labels, read_motion_cache
from mint.muscle_schema import validate_activation_matrix


class ActivationMintDataset(Dataset):
    """Sliding windows: normalized MinT q + 402 muscle activations."""

    def __init__(
        self,
        data_root: str,
        *,
        split: str = "train",
        window_size: int = 64,
        window_stride: int = 16,
        normalize_q: bool = True,
        max_motions: int = 0,
        skip_unlabeled: bool = True,
        skip_zero_placeholders: bool = True,
        zero_atol: float = 1e-8,
    ):
        self.data_root = Path(data_root)
        self.split = str(split)
        self.window_size = int(window_size)
        self.window_stride = max(1, int(window_stride))
        self.normalize_q = bool(normalize_q)
        self.skip_unlabeled = bool(skip_unlabeled)
        self.skip_zero_placeholders = bool(skip_zero_placeholders)
        self.zero_atol = float(zero_atol)

        self.cache_dir = mint_cache_dir(self.data_root)
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(
                f"Missing {self.cache_dir}. Run scripts/preprocess_mint.py first."
            )

        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        if self.normalize_q:
            mean_p, std_p = self.cache_dir / "Mean.npy", self.cache_dir / "Std.npy"
            if not mean_p.is_file() or not std_p.is_file():
                raise FileNotFoundError(
                    f"MinT q stats missing under {self.cache_dir}. "
                    "Run scripts/compute_normalization.py --skeleton mint."
                )
            self.mean = np.load(mean_p).astype(np.float32)
            self.std = np.load(std_p).astype(np.float32)

        ids = load_split_ids(self.data_root, self.split)
        if int(max_motions) > 0:
            ids = ids[: int(max_motions)]

        self._windows: List[Tuple[str, int]] = []
        self.num_motions_seen = 0
        self.num_motions_kept = 0
        self.num_motions_skipped = 0

        for sid in ids:
            npz_path = self.cache_dir / f"{sid}.npz"
            if not npz_path.is_file():
                self.num_motions_skipped += 1
                continue
            self.num_motions_seen += 1
            data = read_motion_cache(npz_path)
            q = data[KEY_Q]
            act = data[KEY_MUSCLE]
            tlen = int(q.shape[0])
            if tlen < self.window_size:
                self.num_motions_skipped += 1
                continue
            if self.skip_unlabeled and not cache_has_labels(npz_path):
                self.num_motions_skipped += 1
                continue
            if self.skip_zero_placeholders and validate_activation_matrix(act, atol=self.zero_atol):
                self.num_motions_skipped += 1
                continue
            if act.shape[1] != MINT_MUSCLE_COUNT:
                self.num_motions_skipped += 1
                continue

            self.num_motions_kept += 1
            for st in range(0, tlen - self.window_size + 1, self.window_stride):
                self._windows.append((str(npz_path), int(st)))

        if not self._windows:
            raise ValueError(
                f"No MinT activation windows for split={self.split!r} under {self.cache_dir}."
            )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, start = self._windows[idx]
        data = read_motion_cache(path)
        n = self.window_size
        q = data[KEY_Q][start : start + n].astype(np.float32)
        act = data[KEY_MUSCLE][start : start + n].astype(np.float32)
        if self.mean is not None and self.std is not None:
            q = (q - self.mean) / np.maximum(self.std, 1e-8)
        return torch.from_numpy(q), torch.from_numpy(act)
