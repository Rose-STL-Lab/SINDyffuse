"""MinT NPZ text-motion dataset for diffusion training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from common.paths import humanml3d_text_dir, mint_cache_dir
from osim.cache_schema import KEY_Q, load_cache_metadata, read_motion_cache


class MintMotionDataset(Dataset):
    """Windowed MinT ``q`` from NPZ cache with text captions."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        window_size: int = 64,
        fps: int = 20,
        normalize: bool = True,
        preload: bool = False,
    ):
        self.data_root = Path(data_root)
        self.split = str(split)
        self.window_size = int(window_size)
        self.fps = int(fps)
        self.normalize = bool(normalize)
        self.preload = bool(preload)

        split_file = self.data_root / (
            "val.txt" if self.split in {"val", "validation"} else f"{self.split}.txt"
        )
        self.cache_dir = mint_cache_dir(self.data_root)
        self.text_dir = humanml3d_text_dir(self.data_root)
        if not split_file.is_file():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(
                f"Missing {self.cache_dir}. Run scripts/preprocess_mint.py first."
            )

        meta = load_cache_metadata(self.cache_dir)
        self.feature_dim = int(meta.get("num_dofs", 0))

        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        if self.normalize:
            mean_p, std_p = self.cache_dir / "Mean.npy", self.cache_dir / "Std.npy"
            if not mean_p.is_file() or not std_p.is_file():
                raise FileNotFoundError(
                    f"MinT q stats missing at {self.data_root}. "
                    "Run scripts/compute_normalization.py --skeleton mint."
                )
            self.mean = np.load(mean_p).astype(np.float32)
            self.std = np.load(std_p).astype(np.float32)

        ids = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        self._motion_ids: List[str] = []
        self._captions: List[str] = []
        self._start_frames: List[int] = []
        self._npz_paths: List[str] = []
        self._preload: Dict[str, np.ndarray] = {}

        for sid in ids:
            npz_path = self.cache_dir / f"{sid}.npz"
            text_path = self.text_dir / f"{sid}.txt"
            if not npz_path.is_file() or not text_path.is_file():
                continue
            data = read_motion_cache(npz_path)
            q = data[KEY_Q]
            tlen = int(q.shape[0])
            if tlen < self.window_size:
                continue
            if self.feature_dim <= 0:
                self.feature_dim = int(q.shape[1])
            if self.preload:
                self._preload[str(npz_path)] = q.astype(np.float32)

            for line in text_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split("#")
                caption = parts[0].strip()
                start, end = 0, tlen
                if len(parts) > 1:
                    try:
                        st_e = parts[1].strip().split()
                        if len(st_e) >= 2:
                            start = int(float(st_e[0]))
                            end = int(float(st_e[1]))
                    except ValueError:
                        pass
                start = max(0, min(start, tlen - 1))
                end = max(start + 1, min(end, tlen))
                seg_len = end - start
                if seg_len < self.window_size:
                    continue
                for st in range(start, end - self.window_size + 1, max(1, self.window_size // 4)):
                    self._motion_ids.append(sid)
                    self._captions.append(caption)
                    self._start_frames.append(int(st))
                    self._npz_paths.append(str(npz_path))

        if not self._motion_ids:
            raise ValueError(f"No MinT windows for split={self.split!r}")

    def __len__(self) -> int:
        return len(self._motion_ids)

    def _read_q(self, path: str, start: int, n: int) -> np.ndarray:
        if path in self._preload:
            q = self._preload[path]
        else:
            q = read_motion_cache(path)[KEY_Q]
        seg = q[start : start + n].astype(np.float32)
        if self.mean is not None and self.std is not None:
            seg = (seg - self.mean) / np.maximum(self.std, 1e-8)
        return seg

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        path = self._npz_paths[idx]
        start = self._start_frames[idx]
        q = self._read_q(path, start, self.window_size)
        return {
            "motion": torch.from_numpy(q),
            "caption": self._captions[idx],
            "motion_id": self._motion_ids[idx],
        }
