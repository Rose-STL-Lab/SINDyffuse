"""SINDyffuse-specific biomechanical metrics for guidance ablations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common.paths import mint_cache_dir, resolve_data_root
from osim.cache_schema import KEY_Q, read_motion_cache
from osim.physics import bio_from_q_torch


@dataclass
class BiomechEvalResults:
    bio_mse_mean: float
    bio_mse_std: float
    joint_limit_violation_mean: float
    contact_gap_mean: float
    num_motions: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _denormalize_q(q_norm: np.ndarray, cache_dir: Path) -> np.ndarray:
    mean = np.load(cache_dir / "Mean.npy").astype(np.float32)
    std = np.load(cache_dir / "Std.npy").astype(np.float32)
    return q_norm * std + mean


def compute_bio_mse(pred_q: np.ndarray, ref_q: np.ndarray, *, fps: float = 20.0) -> float:
    pred = torch.from_numpy(pred_q).float().unsqueeze(0)
    ref = torch.from_numpy(ref_q).float().unsqueeze(0)
    pred_bio = bio_from_q_torch(pred, fps=fps)
    ref_bio = bio_from_q_torch(ref, fps=fps)
    return float(torch.mean((pred_bio - ref_bio) ** 2).item())


def evaluate_biomech_from_generated(
    generated_dir: Path,
    *,
    data_root: str,
    fps: float = 20.0,
) -> BiomechEvalResults:
    root = Path(resolve_data_root(data_root))
    cache_dir = mint_cache_dir(root)
    generated_dir = generated_dir.expanduser().resolve()

    bio_mses: list[float] = []
    joint_limits: list[float] = []
    contact_gaps: list[float] = []

    for path in sorted(generated_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        motion_id = str(data.get("motion_id", path.stem))
        q_norm = np.asarray(data["motion"], dtype=np.float32)
        ref_path = cache_dir / f"{motion_id}.npz"
        if not ref_path.is_file():
            continue
        ref_q = read_motion_cache(ref_path)[KEY_Q].astype(np.float32)
        n = min(len(q_norm), len(ref_q))
        q = _denormalize_q(q_norm[:n], cache_dir)
        ref = ref_q[:n]
        bio_mses.append(compute_bio_mse(q, ref, fps=fps))

        q_t = torch.from_numpy(q).float().unsqueeze(0)
        bio = bio_from_q_torch(q_t, fps=fps)
        joint_limits.append(float(bio[..., 7].mean().item()))
        contact_gaps.append(float(bio[..., 16].mean().item()))

    if not bio_mses:
        raise ValueError(f"No comparable generated motions found in {generated_dir}")

    return BiomechEvalResults(
        bio_mse_mean=float(np.mean(bio_mses)),
        bio_mse_std=float(np.std(bio_mses)),
        joint_limit_violation_mean=float(np.mean(joint_limits)),
        contact_gap_mean=float(np.mean(contact_gaps)),
        num_motions=len(bio_mses),
    )
