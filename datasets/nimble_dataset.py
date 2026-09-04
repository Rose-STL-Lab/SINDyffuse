from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from common.paths import humanml3d_text_dir, lai_cache_dir
from datasets.lai_cache import list_motion_ids, mean_path, motion_npz_path, read_motion_npz, std_path, write_cache_meta
from datasets.splits import load_split_ids
from nimble.lai_coord_map import LAI_NUM_DOFS


def read_q_frames_from_npz(path: str | Path, *, start_frame: int = 0, num_frames: int | None = None, mmap: bool = False) -> np.ndarray:
    data = read_motion_npz(path, mmap=mmap)
    q = data['q']
    t_len = int(q.shape[0])
    st = max(0, int(start_frame))
    if num_frames is None:
        return np.asarray(q[st:], dtype=np.float32)
    ed = min(t_len, st + int(num_frames))
    if ed <= st:
        raise RuntimeError(f'Empty segment start={start_frame} n={num_frames} in {path}')
    return np.asarray(q[st:ed], dtype=np.float32)


def read_q_segment(npz_path: str, trial: int = 0, seg_start: Optional[int] = None, seg_end: Optional[int] = None) -> np.ndarray:
    del trial  # single-trial NPZ
    data = read_motion_npz(npz_path)
    q = data['q']
    t_len = int(q.shape[0])
    if seg_start is not None and seg_end is not None:
        st = max(0, int(seg_start))
        ed = min(int(seg_end), t_len)
        if ed <= st:
            raise RuntimeError(f'Empty segment [{seg_start},{seg_end}) in {npz_path}')
        return np.asarray(q[st:ed], dtype=np.float32)
    return np.asarray(q, dtype=np.float32)


# Backward-compatible name used by older callers
def read_q_frames(subj_or_path, trial: int = 0, start_frame: int = 0, num_frames: int = 0, *, kin: int | None = None) -> np.ndarray:
    del trial, kin
    if isinstance(subj_or_path, (str, Path)):
        return read_q_frames_from_npz(subj_or_path, start_frame=start_frame, num_frames=num_frames or None)
    raise TypeError('read_q_frames now expects an NPZ path; SubjectOnDisk is no longer supported')


def _resolve_stats_motion_ids(root: Path, cache_dir: Path, *, split: str = 'train', splits: Sequence[str] | None = None) -> tuple[list[str], str]:
    if splits:
        ids: list[str] = []
        for sp in splits:
            ids.extend(load_split_ids(root, str(sp)))
        label = '+'.join((str(sp) for sp in splits))
        return (sorted(set(ids)), label)
    return (load_split_ids(root, split), str(split))


def _accumulate_q_stats(cache_dir: Path, ids: list[str], *, max_frames_per_motion: int = 0) -> tuple[np.ndarray, np.ndarray, int, int]:
    sum_q: np.ndarray | None = None
    sum_sq: np.ndarray | None = None
    count = 0
    num_dofs: int | None = None
    for sid in ids:
        p = motion_npz_path(cache_dir, sid)
        if not p.is_file():
            continue
        try:
            data = read_motion_npz(p, mmap=True)
        except Exception:
            continue
        q = np.asarray(data['q'], dtype=np.float64)
        n_read = int(q.shape[0])
        if int(max_frames_per_motion) > 0:
            n_read = min(n_read, int(max_frames_per_motion))
        q = q[:n_read]
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
        raise RuntimeError(f'No usable NPZ motions under {cache_dir}')
    return (sum_q, sum_sq, count, num_dofs)


def compute_nimble_normalization_stats(
    data_root: str | Path,
    *,
    split: str = 'train',
    splits: Sequence[str] | None = None,
    max_motions: int = 0,
    max_frames_per_motion: int = 0,
) -> dict:
    root = Path(data_root).expanduser().resolve()
    cache_dir = lai_cache_dir(root)
    ids, stats_label = _resolve_stats_motion_ids(root, cache_dir, split=split, splits=splits)
    if int(max_motions) > 0:
        ids = ids[: int(max_motions)]
    sum_q, sum_sq, count, num_dofs = _accumulate_q_stats(cache_dir, ids, max_frames_per_motion=max_frames_per_motion)
    mean = (sum_q / float(count)).astype(np.float64)
    var = np.maximum(sum_sq / float(count) - mean * mean, 0.0)
    std = np.sqrt(var).astype(np.float64)
    std = np.maximum(std, 1e-8)
    np.save(mean_path(cache_dir), mean.astype(np.float32))
    np.save(std_path(cache_dir), std.astype(np.float32))
    write_cache_meta(cache_dir)
    meta = {
        'num_dofs': int(num_dofs),
        'stats_split': str(stats_label),
        'stats_frame_count': int(count),
        'feature_type': 'lai_q',
        'cache': 'lai_cache',
        'mean_path': str(mean_path(cache_dir)),
        'std_path': str(std_path(cache_dir)),
    }
    (cache_dir / 'normalization_meta.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    return meta


class NimbleDataset(Dataset):
    """Lai NPZ motion windows (kept class name for config compatibility)."""

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        window_size: int = 64,
        fps: int = 20,
        normalize: bool = True,
        preload: bool = False,
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = str(split)
        self.window_size = int(window_size)
        self.fps = int(fps)
        self.normalize = bool(normalize)
        self.preload = bool(preload)
        self.cache_dir = lai_cache_dir(self.data_root)
        self.ids = load_split_ids(self.data_root, self.split)
        self.ids = [sid for sid in self.ids if motion_npz_path(self.cache_dir, sid).is_file()]
        self.feature_dim = int(LAI_NUM_DOFS)
        self.mean = np.zeros((self.feature_dim,), dtype=np.float32)
        self.std = np.ones((self.feature_dim,), dtype=np.float32)
        if self.normalize:
            mp, sp = mean_path(self.cache_dir), std_path(self.cache_dir)
            if mp.is_file() and sp.is_file():
                self.mean = np.load(mp).astype(np.float32).reshape(-1)
                self.std = np.load(sp).astype(np.float32).reshape(-1)
                if self.mean.shape[0] != self.feature_dim or self.std.shape[0] != self.feature_dim:
                    raise ValueError(f'Mean/Std dim {self.mean.shape}/{self.std.shape} != {self.feature_dim}')
        self._index: List[Tuple[str, int]] = []
        self._q_cache: Dict[str, np.ndarray] = {}
        self._lengths: Dict[str, int] = {}
        for sid in self.ids:
            data = read_motion_npz(motion_npz_path(self.cache_dir, sid), mmap=not self.preload)
            t_len = int(data['num_frames'])
            self._lengths[sid] = t_len
            if self.preload:
                self._q_cache[sid] = np.asarray(data['q'], dtype=np.float32)
            max_start = t_len - self.window_size
            if max_start < 0:
                continue
            for st in range(0, max_start + 1, max(1, self.window_size // 4) if self.window_size > 0 else 1):
                self._index.append((sid, st))
            if max_start >= 0 and (sid, max_start) not in self._index:
                self._index.append((sid, max_start))
        self.text_dir = humanml3d_text_dir(self.data_root)

    def __len__(self) -> int:
        return len(self._index)

    def _load_q(self, sid: str) -> np.ndarray:
        if sid in self._q_cache:
            return self._q_cache[sid]
        q = read_motion_npz(motion_npz_path(self.cache_dir, sid), mmap=True)['q']
        return np.asarray(q, dtype=np.float32)

    def __getitem__(self, idx: int):
        sid, st = self._index[idx]
        q = self._load_q(sid)
        window = q[st : st + self.window_size]
        if window.shape[0] < self.window_size:
            pad = np.zeros((self.window_size - window.shape[0], window.shape[1]), dtype=np.float32)
            window = np.concatenate([window, pad], axis=0)
        if self.normalize:
            window = (window - self.mean) / self.std
        caption = ''
        text_path = self.text_dir / f'{sid}.txt'
        if text_path.is_file():
            lines = text_path.read_text(encoding='utf-8', errors='ignore').strip().splitlines()
            if lines:
                caption = lines[0].split('#')[0].strip()
        return {
            'motion': torch.from_numpy(window.astype(np.float32)),
            'caption': caption,
            'motion_id': sid,
            'start': int(st),
        }


# Alias
LaiDataset = NimbleDataset
