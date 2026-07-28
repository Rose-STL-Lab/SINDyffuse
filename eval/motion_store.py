"""Save and load generated motion results for evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.io import load_json, save_json


@dataclass
class GeneratedMotionRecord:
    sample_id: str
    motion_id: str
    caption: str
    motion: np.ndarray
    length: int
    format: str = "hml263"
    variant: str = ""
    seed: int = 0
    metadata: dict[str, Any] | None = None


def save_motion_npy(path: Path, motion: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, motion.astype(np.float32))


def load_motion_npy(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def save_generated_record(path: Path, record: GeneratedMotionRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_id": record.sample_id,
        "motion_id": record.motion_id,
        "caption": record.caption,
        "length": record.length,
        "format": record.format,
        "variant": record.variant,
        "seed": record.seed,
        "metadata": record.metadata or {},
    }
    if record.format == "hml263":
        np.savez_compressed(path.with_suffix(".npz"), motion=record.motion.astype(np.float32), **payload)
    else:
        np.savez_compressed(path.with_suffix(".npz"), motion=record.motion.astype(np.float32), **payload)


def load_generated_records(root: Path, *, format: str = "hml263") -> dict[str, np.ndarray]:
    root = root.expanduser().resolve()
    lookup: dict[str, np.ndarray] = {}
    if not root.is_dir():
        return lookup

    for path in sorted(root.glob("*.npy")):
        motion_id = path.stem
        lookup[motion_id] = load_motion_npy(path)

    for path in sorted(root.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        motion_id = str(data.get("motion_id", path.stem))
        motion = np.asarray(data["motion"], dtype=np.float32)
        lookup[motion_id] = motion
    return lookup


def write_manifest(path: Path, records: list[GeneratedMotionRecord]) -> None:
    payload = {
        "records": [
            {
                **asdict(record),
                "motion": None,
            }
            for record in records
        ]
    }
    save_json(str(path), payload)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = load_json(str(path))
    return list(payload.get("records", []))


def method_motion_lookup(method_dir: Path) -> dict[str, np.ndarray]:
    return load_generated_records(method_dir)
