"""HumanML3D dataset loader (joint vecs + text captions)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class CaptionSample:
    sample_id: str
    caption: str
    start_frame: Optional[int]
    end_frame: Optional[int]


class HumanML3DTextMotionDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train", window_size: int = 64, fps: int = 20, normalize: bool = True):
        self.data_root = Path(data_root)
        self.split = str(split)
        self.window_size = int(window_size)
        self.fps = int(fps)
        self.normalize = bool(normalize)

        split_file = self.data_root / ("val.txt" if self.split in {"val", "validation"} else f"{self.split}.txt")
        self.motion_dir = self.data_root / "new_joint_vecs"
        self.text_dir = self.data_root / "texts"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")

        self.mean = None
        self.std = None
        if self.normalize:
            self.mean = np.load(self.data_root / "Mean.npy").astype(np.float32)
            self.std = np.load(self.data_root / "Std.npy").astype(np.float32)

        ids = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.samples: List[CaptionSample] = []
        self.motion_cache: Dict[str, np.ndarray] = {}
        self.feature_dim = 0
        for sid in ids:
            motion_path = self.motion_dir / f"{sid}.npy"
            text_path = self.text_dir / f"{sid}.txt"
            if not motion_path.exists() or not text_path.exists():
                continue
            motion = np.load(motion_path).astype(np.float32)
            if motion.ndim != 2:
                continue
            self.motion_cache[sid] = motion
            self.feature_dim = int(motion.shape[1])
            for line in text_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split("#")
                if len(parts) != 4:
                    continue
                cap = parts[0].strip()
                if not cap:
                    continue
                try:
                    f_tag = float(parts[2])
                    t_tag = float(parts[3])
                except Exception:
                    f_tag = 0.0
                    t_tag = 0.0
                if f_tag == 0.0 and t_tag == 0.0:
                    self.samples.append(CaptionSample(sid, cap, None, None))
                else:
                    st = int(max(0.0, f_tag) * self.fps)
                    ed = int(max(0.0, t_tag) * self.fps)
                    if ed > st:
                        self.samples.append(CaptionSample(sid, cap, st, ed))
        if not self.samples:
            raise ValueError("No valid HumanML3D samples found")

    def __len__(self) -> int:
        return len(self.samples)

    def _crop_or_pad(self, arr: np.ndarray) -> np.ndarray:
        t = int(arr.shape[0])
        if t >= self.window_size:
            st = np.random.randint(0, t - self.window_size + 1)
            return arr[st : st + self.window_size]
        rep = (self.window_size + t - 1) // max(t, 1)
        return np.tile(arr, (rep, 1))[: self.window_size]

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        motion = self.motion_cache[s.sample_id]
        clip = motion
        if s.start_frame is not None and s.end_frame is not None:
            st = min(max(0, int(s.start_frame)), motion.shape[0] - 1)
            ed = min(max(st + 1, int(s.end_frame)), motion.shape[0])
            clip = motion[st:ed]
        if clip.shape[0] < 2:
            clip = motion
        clip = self._crop_or_pad(clip)
        if self.normalize:
            clip = (clip - self.mean) / np.clip(self.std, 1e-8, None)
        return {"motion": torch.tensor(clip, dtype=torch.float32), "caption": s.caption, "sample_id": s.sample_id}
