"""HumanML3D split lists and B3D kinematics pass helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_SPLIT_FILES = ("train.txt", "val.txt", "test.txt")


def load_split_ids(root: Path, split: str) -> list[str]:
    name = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}.get(split.strip().lower())
    if name is None:
        raise ValueError(f"Unknown split: {split}")
    p = root / name
    if not p.is_file():
        raise FileNotFoundError(f"Missing split file: {p}")
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def all_motion_ids(root: Path) -> list[str]:
    """All HumanML3D motion IDs (union of train, val, and test lists)."""
    ids: list[str] = []
    for name in _SPLIT_FILES:
        p = root / name
        if not p.is_file():
            raise FileNotFoundError(f"Missing split file: {p}")
        ids.extend(ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())
    return sorted(set(ids))


def shard_motion_ids(ids: list[str], shard_index: int, num_shards: int) -> list[str]:
    """Strided subset for distributed preprocess (``ids[shard_index::num_shards]``)."""
    n = int(num_shards)
    if n <= 1:
        return ids
    i = int(shard_index)
    if i < 0 or i >= n:
        raise ValueError(f"shard_index must be in [0, {n}), got {i}")
    return ids[i::n]


def kinematics_pass_index(subj: Any, trial: int) -> int:
    import nimblephysics as nimble

    n = int(subj.getTrialNumProcessingPasses(trial))
    for i in range(n):
        ptype = str(subj.getProcessingPassType(i)).upper()
        if "KINEMATICS" in ptype:
            return i
    return 0
