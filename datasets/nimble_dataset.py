"""Nimble B3D text-motion dataset (generalized coordinates + captions)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import nimblephysics as nimble

from common.paths import nimble_b3d_dir
from nimble.b3d_schema import metadata_custom_values_block
from datasets.splits import kinematics_pass_index, load_split_ids

# Cache key: (b3d_path, trial, seg_start, seg_end) with None for full trial.
_QCacheKey = Tuple[str, int, Optional[int], Optional[int]]


def read_q_frames(
    subj: nimble.biomechanics.SubjectOnDisk,
    trial: int,
    start_frame: int,
    num_frames: int,
    *,
    kin: int | None = None,
) -> np.ndarray:
    """Read ``num_frames`` of ``q`` from a B3D subject starting at ``start_frame``."""
    if kin is None:
        kin = kinematics_pass_index(subj, trial)
    frames = subj.readFrames(
        trial=trial,
        startFrame=int(start_frame),
        numFramesToRead=int(num_frames),
        includeSensorData=False,
        includeProcessingPasses=True,
    )
    rows: List[np.ndarray] = []
    for fr in frames:
        rows.append(np.asarray(fr.processingPasses[kin].pos, dtype=np.float32).reshape(-1))
    if not rows:
        raise RuntimeError(f"Empty B3D read trial={trial} start={start_frame} n={num_frames}")
    return np.stack(rows, axis=0).astype(np.float32)


def read_q_segment(
    b3d_path: str,
    trial: int = 0,
    seg_start: Optional[int] = None,
    seg_end: Optional[int] = None,
) -> np.ndarray:
    """Load a contiguous ``q`` trajectory (full trial or ``[seg_start:seg_end)``)."""
    subj = nimble.biomechanics.SubjectOnDisk(str(b3d_path))
    tlen = int(subj.getTrialLength(trial))
    if tlen < 1:
        raise RuntimeError(f"Empty trial in {b3d_path}")
    kin = kinematics_pass_index(subj, trial)
    if seg_start is not None and seg_end is not None:
        st = max(0, int(seg_start))
        ed = min(int(seg_end), tlen)
        n = max(0, ed - st)
        if n < 1:
            raise RuntimeError(f"Empty segment [{seg_start},{seg_end}) in {b3d_path}")
        return read_q_frames(subj, trial, st, n, kin=kin)
    return read_q_frames(subj, trial, 0, tlen, kin=kin)


def _resolve_stats_motion_ids(
    root: Path,
    cache_dir: Path,
    *,
    split: str = "train",
    splits: Sequence[str] | None = None,
) -> tuple[list[str], str]:
    if splits:
        ids: list[str] = []
        for sp in splits:
            ids.extend(load_split_ids(root, str(sp)))
        label = "+".join(str(sp) for sp in splits)
        return sorted(set(ids)), label
    return load_split_ids(root, split), str(split)


def _accumulate_q_stats(
    cache_dir: Path,
    ids: list[str],
    *,
    max_frames_per_motion: int = 0,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    sum_q: np.ndarray | None = None
    sum_sq: np.ndarray | None = None
    count = 0
    num_dofs: int | None = None

    for sid in ids:
        p = cache_dir / f"{sid}.b3d"
        if not p.is_file():
            continue
        subj = nimble.biomechanics.SubjectOnDisk(str(p))
        trial = 0
        tlen = int(subj.getTrialLength(trial))
        if tlen < 1:
            continue
        kin = kinematics_pass_index(subj, trial)
        n_read = tlen
        if int(max_frames_per_motion) > 0:
            n_read = min(n_read, int(max_frames_per_motion))
        frames = subj.readFrames(
            trial=trial,
            startFrame=0,
            numFramesToRead=n_read,
            includeSensorData=False,
            includeProcessingPasses=True,
        )
        for fr in frames:
            pos = np.asarray(fr.processingPasses[kin].pos, dtype=np.float64).reshape(-1)
            if num_dofs is None:
                num_dofs = int(pos.shape[0])
                sum_q = np.zeros(num_dofs, dtype=np.float64)
                sum_sq = np.zeros(num_dofs, dtype=np.float64)
            if int(pos.shape[0]) != num_dofs:
                continue
            sum_q += pos
            sum_sq += pos * pos
            count += 1
        del subj

    if count < 1 or sum_q is None or sum_sq is None or num_dofs is None:
        raise RuntimeError("No frames accumulated for Nimble q statistics")
    return sum_q, sum_sq, int(count), int(num_dofs)


def compute_nimble_normalization_stats(
    data_root: str | Path,
    *,
    split: str = "train",
    splits: Sequence[str] | None = None,
    max_motions: int = 0,
    max_frames_per_motion: int = 0,
) -> dict:
    """Accumulate q mean/std over a split; write ``Mean.npy`` and ``Std.npy`` under the B3D cache."""
    root = Path(data_root).expanduser().resolve()
    cache_dir = nimble_b3d_dir(root)
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Missing Nimble B3D cache: {cache_dir}")

    ids, stats_label = _resolve_stats_motion_ids(root, cache_dir, split=split, splits=splits)
    if int(max_motions) > 0:
        ids = ids[: int(max_motions)]

    try:
        sum_q, sum_sq, count, num_dofs = _accumulate_q_stats(
            cache_dir, ids, max_frames_per_motion=max_frames_per_motion
        )
    except RuntimeError:
        cached = sorted(p.stem for p in cache_dir.glob("*.b3d"))
        if not cached:
            raise RuntimeError(f"No frames accumulated for Nimble q statistics under {root}") from None
        sum_q, sum_sq, count, num_dofs = _accumulate_q_stats(
            cache_dir, cached, max_frames_per_motion=max_frames_per_motion
        )
        stats_label = "cache"

    mean = (sum_q / float(count)).astype(np.float32)
    var = np.maximum(sum_sq / float(count) - mean.astype(np.float64) ** 2, 1e-12)
    std = np.sqrt(var).astype(np.float32)
    std = np.clip(std, 1e-6, None)

    np.save(cache_dir / "Mean.npy", mean)
    np.save(cache_dir / "Std.npy", std)

    meta_path = cache_dir / "metadata.json"
    meta: dict = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    meta.update(
        {
            "num_dofs": int(num_dofs),
            "stats_split": str(stats_label),
            "stats_frame_count": int(count),
            "feature_type": "nimble_q",
            "b3d_custom_values": metadata_custom_values_block(),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "num_dofs": int(num_dofs),
        "stats_split": str(stats_label),
        "stats_frame_count": int(count),
        "mean_path": str(cache_dir / "Mean.npy"),
        "std_path": str(cache_dir / "Std.npy"),
    }


class NimbleDataset(Dataset):
    """Load windowed Nimble ``q`` from per-motion B3D files with text captions."""

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
        self.b3d_dir = nimble_b3d_dir(self.data_root)
        self.text_dir = self.data_root / "texts"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        if not self.b3d_dir.is_dir():
            raise FileNotFoundError(
                f"Missing {self.b3d_dir}. Run preprocess_nimble.py first."
            )

        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        if self.normalize:
            mean_p, std_p = self.b3d_dir / "Mean.npy", self.b3d_dir / "Std.npy"
            if not mean_p.is_file() or not std_p.is_file():
                raise FileNotFoundError(
                    f"Nimble Q stats missing at {self.data_root}. "
                    f"Run preprocess_nimble.py first (writes Mean.npy / Std.npy)."
                )
            self.mean = np.load(mean_p).astype(np.float32)
            self.std = np.load(std_p).astype(np.float32)

        ids = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        self._motion_ids: List[str] = []
        self._captions: List[str] = []
        self._start_frames: List[Optional[int]] = []
        self._end_frames: List[Optional[int]] = []
        self._b3d_paths: List[str] = []
        self._trials: List[int] = []
        self.feature_dim = 0
        self._trial_lengths: Dict[str, int] = {}

        for sid in ids:
            b3d_path = self.b3d_dir / f"{sid}.b3d"
            text_path = self.text_dir / f"{sid}.txt"
            if not b3d_path.is_file() or not text_path.is_file():
                continue
            subj = nimble.biomechanics.SubjectOnDisk(str(b3d_path))
            trial = 0
            tlen = int(subj.getTrialLength(trial))
            if tlen < 2:
                continue
            ndof = int(subj.getNumDofs())
            self.feature_dim = ndof
            self._trial_lengths[sid] = tlen

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
                    self._append_index(sid, cap, None, None, str(b3d_path), trial)
                else:
                    st = int(max(0.0, f_tag) * self.fps)
                    ed = int(max(0.0, t_tag) * self.fps)
                    if ed > st:
                        self._append_index(sid, cap, st, ed, str(b3d_path), trial)

        if not self._motion_ids:
            raise ValueError("No valid Nimble B3D caption entries found")

        self._subj_cache: Dict[str, nimble.biomechanics.SubjectOnDisk] = {}
        self._q_cache: Dict[_QCacheKey, np.ndarray] = {}
        if self.preload:
            self._preload_q_trajectories()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_subj_cache"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if not hasattr(self, "_subj_cache"):
            self._subj_cache = {}
        if not hasattr(self, "_q_cache"):
            self._q_cache = {}
        if not hasattr(self, "preload"):
            self.preload = False

    def _q_cache_key(self, idx: int) -> _QCacheKey:
        return (
            self._b3d_paths[idx],
            self._trials[idx],
            self._start_frames[idx],
            self._end_frames[idx],
        )

    def _preload_q_trajectories(self) -> None:
        keys = {self._q_cache_key(i) for i in range(len(self._motion_ids))}
        print(f"[nimble/data] preloading {len(keys)} q trajectories …", flush=True)
        for i, key in enumerate(sorted(keys, key=lambda k: (k[0], k[1], k[2] or 0, k[3] or 0))):
            path, trial, seg_st, seg_ed = key
            self._q_cache[key] = read_q_segment(path, trial=trial, seg_start=seg_st, seg_end=seg_ed)
            if (i + 1) % 500 == 0 or (i + 1) == len(keys):
                print(f"[nimble/data] preloaded {i + 1}/{len(keys)} trajectories", flush=True)

    def _append_index(
        self,
        motion_id: str,
        caption: str,
        start_frame: Optional[int],
        end_frame: Optional[int],
        b3d_path: str,
        trial: int,
    ) -> None:
        self._motion_ids.append(motion_id)
        self._captions.append(caption)
        self._start_frames.append(start_frame)
        self._end_frames.append(end_frame)
        self._b3d_paths.append(b3d_path)
        self._trials.append(int(trial))

    def _subject(self, path: str) -> nimble.biomechanics.SubjectOnDisk:
        if path not in self._subj_cache:
            self._subj_cache[path] = nimble.biomechanics.SubjectOnDisk(path)
        return self._subj_cache[path]

    def __len__(self) -> int:
        return len(self._motion_ids)

    def _crop_start(self, t_total: int, seg_start: Optional[int], seg_end: Optional[int]) -> int:
        if seg_start is not None and seg_end is not None:
            seg_len = max(1, int(seg_end) - int(seg_start))
            if seg_len >= self.window_size:
                st = int(seg_start) + np.random.randint(0, seg_len - self.window_size + 1)
                return min(st, max(0, t_total - self.window_size))
        if t_total >= self.window_size:
            return int(np.random.randint(0, t_total - self.window_size + 1))
        return 0

    def _pad_window(self, window: np.ndarray) -> np.ndarray:
        if window.shape[0] >= self.window_size:
            return window[: self.window_size]
        rep = (self.window_size + window.shape[0] - 1) // max(window.shape[0], 1)
        return np.tile(window, (rep, 1))[: self.window_size]

    def _normalize_window(self, window: np.ndarray) -> np.ndarray:
        if self.normalize and self.mean is not None and self.std is not None:
            return (window - self.mean) / np.clip(self.std, 1e-8, None)
        return window

    def _read_window_on_demand(self, idx: int) -> np.ndarray:
        subj = self._subject(self._b3d_paths[idx])
        trial = self._trials[idx]
        t_total = int(subj.getTrialLength(trial))
        kin = kinematics_pass_index(subj, trial)
        st = self._crop_start(t_total, self._start_frames[idx], self._end_frames[idx])
        window = read_q_frames(subj, trial, st, self.window_size, kin=kin)
        return self._pad_window(window)

    def _read_window_from_cache(self, idx: int) -> np.ndarray:
        arr = self._q_cache[self._q_cache_key(idx)]
        st = self._crop_start(int(arr.shape[0]), None, None)
        window = arr[st : st + self.window_size]
        return self._pad_window(window)

    def __getitem__(self, idx: int):
        if self.preload:
            window = self._read_window_from_cache(idx)
        else:
            window = self._read_window_on_demand(idx)
        window = self._normalize_window(window.astype(np.float32))

        return {
            "motion": torch.tensor(window, dtype=torch.float32),
            "caption": self._captions[idx],
            "motion_id": self._motion_ids[idx],
        }
