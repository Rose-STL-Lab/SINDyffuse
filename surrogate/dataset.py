"""B3D windows: Rajagopal ``q`` + cached OpenSim muscle activations."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import nimblephysics as nimble

from common.paths import nimble_b3d_dir
from datasets.nimble_dataset import read_q_frames
from datasets.splits import kinematics_pass_index, load_split_ids
from nimble.b3d_io import (
    b3d_has_activation_mask,
    b3d_has_sindyffuse_custom_values,
    read_muscle_activation_mask_frames,
    read_muscle_activations_frames,
)

_WARNED_LEGACY_MASK = False


class ActivationB3DDataset(Dataset):
    """Sliding windows over preprocessed B3D with ``muscle_activations`` custom values."""

    def __init__(
        self,
        data_root: str,
        *,
        split: str = "train",
        window_size: int = 64,
        window_stride: int = 16,
        normalize_q: bool = True,
        max_motions: int = 0,
        min_window_valid_fraction: float = 0.35,
        require_activation_mask: bool = False,
    ):
        self.data_root = Path(data_root)
        self.split = str(split)
        self.window_size = int(window_size)
        self.window_stride = max(1, int(window_stride))
        self.normalize_q = bool(normalize_q)
        self.min_window_valid_fraction = float(min_window_valid_fraction)
        self.require_activation_mask = bool(require_activation_mask)

        self.b3d_dir = nimble_b3d_dir(self.data_root)
        if not self.b3d_dir.is_dir():
            raise FileNotFoundError(
                f"Missing {self.b3d_dir}. Run preprocess_nimble.py first."
            )

        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        if self.normalize_q:
            mean_p, std_p = self.b3d_dir / "Mean.npy", self.b3d_dir / "Std.npy"
            if not mean_p.is_file() or not std_p.is_file():
                raise FileNotFoundError(
                    f"Nimble Q stats missing at {self.data_root}. "
                    f"Run preprocess_nimble.py first (writes Mean.npy / Std.npy)."
                )
            self.mean = np.load(mean_p).astype(np.float32)
            self.std = np.load(std_p).astype(np.float32)

        ids = load_split_ids(self.data_root, self.split)
        if int(max_motions) > 0:
            ids = ids[: int(max_motions)]

        self._windows: List[Tuple[str, int, int]] = []
        self._trial_mask_cache: dict[str, np.ndarray | None] = {}
        global _WARNED_LEGACY_MASK
        for sid in ids:
            b3d_path = self.b3d_dir / f"{sid}.b3d"
            if not b3d_path.is_file():
                continue
            subj = nimble.biomechanics.SubjectOnDisk(str(b3d_path))
            if not b3d_has_sindyffuse_custom_values(subj):
                continue
            has_mask = b3d_has_activation_mask(subj)
            if self.require_activation_mask and not has_mask:
                continue
            if not has_mask and not _WARNED_LEGACY_MASK:
                warnings.warn(
                    f"B3D under {self.b3d_dir} missing muscle_activation_mask; "
                    "training with all-ones mask. Re-run preprocess_nimble.py.",
                    stacklevel=2,
                )
                _WARNED_LEGACY_MASK = True

            tlen = int(subj.getTrialLength(0))
            if tlen < self.window_size:
                continue

            trial_mask: np.ndarray | None = None
            if has_mask:
                trial_mask = read_muscle_activation_mask_frames(subj, 0, 0, tlen)
                self._trial_mask_cache[str(b3d_path)] = trial_mask
            else:
                self._trial_mask_cache[str(b3d_path)] = None

            for st in range(0, tlen - self.window_size + 1, self.window_stride):
                if trial_mask is not None and self.min_window_valid_fraction > 0.0:
                    frac = float(trial_mask[st : st + self.window_size].mean())
                    if frac < self.min_window_valid_fraction:
                        continue
                self._windows.append((str(b3d_path), 0, int(st)))

        if not self._windows:
            raise ValueError(
                f"No activation windows for split={self.split!r} under {self.b3d_dir}. "
                f"Re-run preprocess_nimble.py to embed muscle_activations."
            )

        self._subj_cache: dict[str, nimble.biomechanics.SubjectOnDisk] = {}

    def __len__(self) -> int:
        return len(self._windows)

    def _get_subj(self, path: str) -> nimble.biomechanics.SubjectOnDisk:
        if path not in self._subj_cache:
            self._subj_cache[path] = nimble.biomechanics.SubjectOnDisk(path)
        return self._subj_cache[path]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b3d_path, trial, start = self._windows[idx]
        n = self.window_size
        subj = self._get_subj(b3d_path)
        kin = kinematics_pass_index(subj, trial)
        q = read_q_frames(subj, trial, start, n, kin=kin).astype(np.float32)
        act = read_muscle_activations_frames(subj, trial, start, n).astype(np.float32)
        trial_mask = self._trial_mask_cache.get(b3d_path)
        if trial_mask is not None:
            mask = trial_mask[start : start + n].astype(np.float32)
        else:
            mask = np.ones((n,), dtype=np.float32)
        if self.mean is not None and self.std is not None:
            q = (q - self.mean) / np.maximum(self.std, 1e-8)
        return torch.from_numpy(q), torch.from_numpy(act), torch.from_numpy(mask)
