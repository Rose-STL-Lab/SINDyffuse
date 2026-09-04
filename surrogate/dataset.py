from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from common.paths import lai_cache_dir
from datasets.lai_cache import mean_path, motion_npz_path, read_motion_npz, std_path
from datasets.splits import load_split_ids
from nimble.gap_utils import is_nan_placeholder_activations, window_is_valid
from nimble.muscle_b3d import is_zero_placeholder_activations


class ActivationB3DDataset(Dataset):
    """Lai NPZ activation windows (class name kept for config compatibility)."""

    def __init__(
        self,
        data_root: str,
        *,
        split: str = 'train',
        window_size: int = 64,
        window_stride: int = 16,
        normalize_q: bool = True,
        max_motions: int = 0,
        skip_zero_placeholders: bool = True,
        skip_invalid_activations: bool = True,
        zero_atol: float = 1e-08,
        min_valid_fraction: float = 0.95,
    ):
        self.data_root = Path(data_root)
        self.split = str(split)
        self.window_size = int(window_size)
        self.window_stride = max(1, int(window_stride))
        self.normalize_q = bool(normalize_q)
        self.skip_zero_placeholders = bool(skip_zero_placeholders)
        self.skip_invalid_activations = bool(skip_invalid_activations)
        self.min_valid_fraction = float(min_valid_fraction)
        self.zero_atol = float(zero_atol)
        self.cache_dir = lai_cache_dir(self.data_root)
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(f'Missing {self.cache_dir}. Run preprocess pipeline first.')
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        if self.normalize_q:
            mp, sp = mean_path(self.cache_dir), std_path(self.cache_dir)
            if not mp.is_file() or not sp.is_file():
                raise FileNotFoundError(
                    f'Lai Q stats missing under {self.cache_dir}. Run scripts/compute_normalization.py.'
                )
            self.mean = np.load(mp).astype(np.float32)
            self.std = np.load(sp).astype(np.float32)
        ids = load_split_ids(self.data_root, self.split)
        if int(max_motions) > 0:
            ids = ids[: int(max_motions)]
        self._windows: List[Tuple[str, int]] = []
        self.num_motions_seen = 0
        self.num_motions_kept = 0
        self.num_motions_skipped_zero = 0
        self.num_motions_skipped_nan = 0
        self.num_windows_skipped_gap = 0
        self.num_motions_skipped_other = 0
        self.num_motions_skipped_corrupt = 0
        self._cache: dict[str, dict] = {}
        for sid in ids:
            npz_path = motion_npz_path(self.cache_dir, sid)
            if not npz_path.is_file():
                self.num_motions_skipped_other += 1
                continue
            self.num_motions_seen += 1
            try:
                data = read_motion_npz(npz_path)
            except Exception:
                self.num_motions_skipped_corrupt += 1
                continue
            act = data['muscle_activations']
            mask = data['muscle_activation_mask']
            tlen = int(data['num_frames'])
            if tlen < self.window_size:
                self.num_motions_skipped_other += 1
                continue
            if self.skip_invalid_activations and is_nan_placeholder_activations(act):
                self.num_motions_skipped_nan += 1
                continue
            if self.skip_invalid_activations and float(np.sum(mask > 0.5)) <= 0.0:
                self.num_motions_skipped_nan += 1
                continue
            if self.skip_zero_placeholders and is_zero_placeholder_activations(act, atol=self.zero_atol):
                self.num_motions_skipped_zero += 1
                continue
            self.num_motions_kept += 1
            self._cache[str(npz_path)] = data
            for st in range(0, tlen - self.window_size + 1, self.window_stride):
                if self.skip_invalid_activations:
                    act_win = act[st : st + self.window_size]
                    mask_win = mask[st : st + self.window_size]
                    if not window_is_valid(mask_win, act_win, min_valid_fraction=self.min_valid_fraction):
                        self.num_windows_skipped_gap += 1
                        continue
                self._windows.append((str(npz_path), int(st)))
        if not self._windows:
            raise ValueError(
                f'No activation windows for split={self.split!r} under {self.cache_dir}. '
                f'Re-run preprocess (skipped_zero={self.num_motions_skipped_zero}).'
            )

    def __len__(self) -> int:
        return len(self._windows)

    def _get(self, path: str) -> dict:
        if path not in self._cache:
            self._cache[path] = read_motion_npz(path)
        return self._cache[path]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        npz_path, start = self._windows[idx]
        n = self.window_size
        data = self._get(npz_path)
        q = np.asarray(data['q'][start : start + n], dtype=np.float32)
        act = np.asarray(data['muscle_activations'][start : start + n], dtype=np.float32)
        if self.mean is not None and self.std is not None:
            q = (q - self.mean) / np.maximum(self.std, 1e-08)
        return (torch.from_numpy(q), torch.from_numpy(act))


ActivationLaiDataset = ActivationB3DDataset
