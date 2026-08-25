from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
_SPLIT_FILES = ('train.txt', 'val.txt', 'test.txt')

def load_split_ids(root: Path, split: str) -> list[str]:
    name = {'train': 'train.txt', 'val': 'val.txt', 'test': 'test.txt'}.get(split.strip().lower())
    if name is None:
        raise ValueError(f'Unknown split: {split}')
    p = root / name
    if not p.is_file():
        raise FileNotFoundError(f'Missing split file: {p}')
    return [ln.strip() for ln in p.read_text(encoding='utf-8').splitlines() if ln.strip()]

def all_motion_ids(root: Path) -> list[str]:
    ids: list[str] = []
    for name in _SPLIT_FILES:
        p = root / name
        if not p.is_file():
            raise FileNotFoundError(f'Missing split file: {p}')
        ids.extend((ln.strip() for ln in p.read_text(encoding='utf-8').splitlines() if ln.strip()))
    return sorted(set(ids))

def shard_motion_ids(ids: Sequence[str], shard_index: int, num_shards: int, *, lengths: Mapping[str, float] | None=None) -> list[str]:
    """Assign motions to shards via deterministic longest-first bin packing.

    When ``lengths`` is omitted (or uniform), packing reduces to round-robin over
    sorted ids, matching the historical ``ids[i::n]`` stride assignment.
    """
    n = int(num_shards)
    if n <= 1:
        return list(ids)
    i = int(shard_index)
    if i < 0 or i >= n:
        raise ValueError(f'shard_index must be in [0, {n}), got {i}')

    def _weight(sid: str) -> float:
        if lengths is None:
            return 1.0
        try:
            return max(0.0, float(lengths.get(sid, 1.0)))
        except (TypeError, ValueError):
            return 1.0

    ordered = sorted(ids, key=lambda sid: (-_weight(sid), str(sid)))
    bins: list[list[str]] = [[] for _ in range(n)]
    loads = [0.0] * n
    for sid in ordered:
        b = min(range(n), key=lambda j: (loads[j], j))
        bins[b].append(str(sid))
        loads[b] += _weight(sid)
    return sorted(bins[i])

def kinematics_pass_index(subj: Any, trial: int) -> int:
    import nimblephysics as nimble
    n = int(subj.getTrialNumProcessingPasses(trial))
    for i in range(n):
        ptype = str(subj.getProcessingPassType(i)).upper()
        if 'KINEMATICS' in ptype:
            return i
    return 0
