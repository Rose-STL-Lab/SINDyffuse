from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from common.paths import lai_cache_dir
from datasets.lai_cache import mean_path, motion_npz_path, read_motion_npz, std_path
from datasets.splits import load_split_ids
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
        min_window_size: int = 1,
    ):
        self.data_root = Path(data_root)
        self.split = str(split)
        self.window_size = int(window_size)
        self.min_window_size = int(min_window_size)
        if not 1 <= self.min_window_size <= self.window_size:
            raise ValueError('Require 1 <= min_window_size <= window_size')
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
        self._windows: List[Tuple[str, int, int]] = []
        self.num_valid_frames = 0
        self.num_short_windows = 0
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
            if tlen < self.min_window_size:
                self.num_motions_skipped_other += 1
                continue
            # Invalid labels are never training targets, even when legacy flags
            # request relaxed motion filtering. Keep good runs in partial motions.
            valid = (np.asarray(mask) > 0.5) & np.isfinite(act).all(axis=1) & np.isfinite(data['q']).all(axis=1)
            self.num_valid_frames += int(valid.sum())
            if not valid.any():
                self.num_motions_skipped_nan += 1
                continue
            if self.skip_zero_placeholders and is_zero_placeholder_activations(act, atol=self.zero_atol):
                self.num_motions_skipped_zero += 1
                continue
            before = len(self._windows)
            edges = np.diff(np.r_[False, valid, False].astype(np.int8))
            for begin, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if end - begin < self.min_window_size:
                    self.num_windows_skipped_gap += 1
                    continue
                # Cover each run including its tail, without crossing any gap.
                starts = list(range(int(begin), max(int(begin) + 1, int(end) - self.window_size + 1), min(self.window_stride, self.window_size)))
                tail = max(int(begin), int(end) - self.window_size)
                if starts[-1] != tail:
                    starts.append(tail)
                for st in starts:
                    length = min(self.window_size, int(end) - st)
                    self._windows.append((str(npz_path), st, length))
                    self.num_short_windows += int(length < self.window_size)
            if len(self._windows) > before:
                self.num_motions_kept += 1
        if not self._windows:
            raise ValueError(
                f'No activation windows for split={self.split!r} under {self.cache_dir}. '
                f'Re-run preprocess (skipped_zero={self.num_motions_skipped_zero}).'
            )

    def __len__(self) -> int:
        return len(self._windows)

    def _get(self, path: str) -> dict:
        if path not in self._cache:
            # Avoid retaining the entire activation dataset in each loader worker.
            self._cache.clear()
            self._cache[path] = read_motion_npz(path)
        return self._cache[path]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        npz_path, start, n = self._windows[idx]
        data = self._get(npz_path)
        q = np.asarray(data['q'][start : start + n], dtype=np.float32)
        act = np.asarray(data['muscle_activations'][start : start + n], dtype=np.float32)
        if self.mean is not None and self.std is not None:
            q = (q - self.mean) / np.maximum(self.std, 1e-08)
        return (torch.from_numpy(q), torch.from_numpy(act))

def collate_activation_windows(batch):
    """Pad contiguous valid runs; padding is excluded from attention and losses."""
    lengths = [q.shape[0] for q, _ in batch]
    q = torch.nn.utils.rnn.pad_sequence([q for q, _ in batch], batch_first=True)
    act = torch.nn.utils.rnn.pad_sequence([act for _, act in batch], batch_first=True)
    mask = torch.arange(q.shape[1])[None, :] < torch.tensor(lengths)[:, None]
    return q, act, mask


ActivationLaiDataset = ActivationB3DDataset
