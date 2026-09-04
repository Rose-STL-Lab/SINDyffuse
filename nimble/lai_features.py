"""Lightweight OpenSim FK keypoints for SINDy feature packing (no Nimble Rajagopal)."""
from __future__ import annotations
from typing import List, Sequence, Tuple
import numpy as np
from nimble.opensim_ik import body_world_positions
from sindy.features import features_from_keypoints

# Keypoint slots for sindy.features (IDX_PELVIS=0, IDX_FOOT_L=4, IDX_FOOT_R=5)
IDX_PELVIS = 0
IDX_FOOT_L = 4
IDX_FOOT_R = 5


def _kp_index_constants() -> Tuple[int, int, int]:
    return (IDX_PELVIS, IDX_FOOT_L, IDX_FOOT_R)


def keypoints_from_lai_q(q: np.ndarray, *, n_keypoints: int | None=None) -> np.ndarray:
    """Build [T, K, 3] keypoints from Lai cache q using OpenSim FK (pelvis + feet)."""
    arr = np.asarray(q, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f'Expected q [T, ndof], got {arr.shape}')
    idx_pelvis, idx_foot_l, idx_foot_r = _kp_index_constants()
    k = int(n_keypoints) if n_keypoints is not None else max(idx_pelvis, idx_foot_l, idx_foot_r) + 1
    out = np.zeros((arr.shape[0], k, 3), dtype=np.float32)
    step = max(1, int(arr.shape[0] // 256))
    # Dense for short clips; subsample+interp for long
    sample_idx = list(range(0, arr.shape[0], step))
    if sample_idx[-1] != arr.shape[0] - 1:
        sample_idx.append(arr.shape[0] - 1)
    samples = np.zeros((len(sample_idx), 3, 3), dtype=np.float64)
    for si, t in enumerate(sample_idx):
        bodies = body_world_positions(arr[t], ('pelvis', 'calcn_l', 'calcn_r'))
        samples[si] = bodies
    # interpolate
    sample_t = np.asarray(sample_idx, dtype=np.float64)
    all_t = np.arange(arr.shape[0], dtype=np.float64)
    for bi, ki in enumerate((idx_pelvis, idx_foot_l, idx_foot_r)):
        for c in range(3):
            out[:, ki, c] = np.interp(all_t, sample_t, samples[:, bi, c]).astype(np.float32)
    return out


def features_from_lai_q(q: np.ndarray, fps: float) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    kp = keypoints_from_lai_q(q)
    return features_from_keypoints(kp, fps=fps)


def zero_bio_matrix(t_len: int) -> np.ndarray:
    from nimble.channels import BIOMECH_COMPONENT_KEYS
    return np.zeros((int(t_len), len(BIOMECH_COMPONENT_KEYS)), dtype=np.float32)
