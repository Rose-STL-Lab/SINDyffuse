"""B3D windows: Rajagopal ``q`` + cached OpenSim muscle activations."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import nimblephysics as nimble

from common.paths import nimble_b3d_dir
from datasets.nimble_dataset import read_q_frames
from datasets.splits import kinematics_pass_index, load_split_ids
from nimble.b3d_io import (
    b3d_has_muscle_activations,
    b3d_has_sindyffuse_custom_values,
    read_muscle_activations_frames,
)
from nimble.muscle_b3d import is_zero_placeholder_activations


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
        skip_zero_placeholders: bool = True,
        zero_atol: float = 1e-8,
    ):
        self.data_root = Path(data_root)
        self.split = str(split)
        self.window_size = int(window_size)
        self.window_stride = max(1, int(window_stride))
        self.normalize_q = bool(normalize_q)
        self.skip_zero_placeholders = bool(skip_zero_placeholders)
        self.zero_atol = float(zero_atol)

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
                    f"Nimble Q stats missing under {self.b3d_dir}. "
                    f"Run scripts/compute_normalization.py (writes Mean.npy / Std.npy)."
                )
            self.mean = np.load(mean_p).astype(np.float32)
            self.std = np.load(std_p).astype(np.float32)

        ids = load_split_ids(self.data_root, self.split)
        if int(max_motions) > 0:
            ids = ids[: int(max_motions)]

        self._windows: List[Tuple[str, int, int]] = []
        self.num_motions_seen = 0
        self.num_motions_kept = 0
        self.num_motions_skipped_zero = 0
        self.num_motions_skipped_other = 0
        for sid in ids:
            b3d_path = self.b3d_dir / f"{sid}.b3d"
            if not b3d_path.is_file():
                self.num_motions_skipped_other += 1
                continue
            self.num_motions_seen += 1
            subj = nimble.biomechanics.SubjectOnDisk(str(b3d_path))
            if not b3d_has_sindyffuse_custom_values(subj):
                self.num_motions_skipped_other += 1
                continue
            if not b3d_has_muscle_activations(subj):
                self.num_motions_skipped_other += 1
                continue

            tlen = int(subj.getTrialLength(0))
            if tlen < self.window_size:
                self.num_motions_skipped_other += 1
                continue

            if self.skip_zero_placeholders:
                act = read_muscle_activations_frames(subj, 0, 0, tlen)
                if is_zero_placeholder_activations(act, atol=self.zero_atol):
                    self.num_motions_skipped_zero += 1
                    continue

            self.num_motions_kept += 1
            for st in range(0, tlen - self.window_size + 1, self.window_stride):
                self._windows.append((str(b3d_path), 0, int(st)))

        if not self._windows:
            raise ValueError(
                f"No activation windows for split={self.split!r} under {self.b3d_dir}. "
                f"Re-run preprocess_nimble.py to embed muscle_activations "
                f"(skipped_zero={self.num_motions_skipped_zero})."
            )

        self._subj_cache: dict[str, nimble.biomechanics.SubjectOnDisk] = {}
    def __len__(self) -> int:
        return len(self._windows)

    def _get_subj(self, path: str) -> nimble.biomechanics.SubjectOnDisk:
        if path not in self._subj_cache:
            self._subj_cache[path] = nimble.biomechanics.SubjectOnDisk(path)
        return self._subj_cache[path]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        b3d_path, trial, start = self._windows[idx]
        n = self.window_size
        subj = self._get_subj(b3d_path)
        kin = kinematics_pass_index(subj, trial)
        q = read_q_frames(subj, trial, start, n, kin=kin).astype(np.float32)
        act = read_muscle_activations_frames(subj, trial, start, n).astype(np.float32)
        if self.mean is not None and self.std is not None:
            q = (q - self.mean) / np.maximum(self.std, 1e-8)
        return torch.from_numpy(q), torch.from_numpy(act)
