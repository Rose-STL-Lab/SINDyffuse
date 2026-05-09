"""Load/save compressed NPZ feature archives."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load_npz(npz_path: str) -> Dict[str, np.ndarray]:
    p = Path(npz_path)
    with np.load(p, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def npz_info(npz_path: str) -> Dict[str, Any]:
    arrays = load_npz(npz_path)
    return {
        "npz_file": str(npz_path),
        "keys": sorted(arrays.keys()),
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "dtypes": {k: str(v.dtype) for k, v in arrays.items()},
    }


def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    feat_dir = output_dir / "features_v1"
    feat_dir.mkdir(parents=True, exist_ok=True)
    return {"feat_dir": feat_dir}


def save_npz(
    memory_result: Dict[str, Any],
    output_dir: str,
    file_stem: str | None = None,
) -> Dict[str, Any]:
    out = ensure_dirs(Path(output_dir))
    stem = file_stem if file_stem is not None else Path(memory_result.get("source", "in_memory")).stem
    rows: List[Dict[str, Any]] = []
    for tr in memory_result["trials"]:
        safe_trial = str(tr["trial_name"]).replace("/", "_").replace(" ", "_")
        out_path = out["feat_dir"] / f"{stem}__trial{int(tr['trial_idx']):03d}__{safe_trial}.npz"
        np.savez_compressed(out_path, **tr["features"])
        rows.append(
            {
                "trial_idx": int(tr["trial_idx"]),
                "trial_name": str(tr["trial_name"]),
                "output_npz": str(out_path),
                "num_frames": int(tr["num_frames"]),
                "num_dofs": int(tr["num_dofs"]),
                "dt": float(tr["dt"]),
                "has_positions": bool("positions" in tr["features"]),
                "contact_wrench_dim": int(tr["features"]["contact_wrench"].shape[1]),
            }
        )
    return {"source": memory_result.get("source"), "trials": rows}
