"""HumanML3D split lists and B3D kinematics pass helpers."""

from __future__ import annotations

from pathlib import Path

import nimblephysics as nimble


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


def kinematics_pass_index(subj: nimble.biomechanics.SubjectOnDisk, trial: int) -> int:
    n = int(subj.getTrialNumProcessingPasses(trial))
    for i in range(n):
        ptype = str(subj.getProcessingPassType(i)).upper()
        if "KINEMATICS" in ptype:
            return i
    return 0
