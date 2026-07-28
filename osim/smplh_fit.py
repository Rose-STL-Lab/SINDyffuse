"""Fit SMPL-H to HumanML3D 22×3 joint trajectories."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

NUM_HML3D_JOINTS = 22

# HumanML3D / SMPL-H body joint order (first 22 SMPL-H joints).
HML_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)


@dataclass
class SmplhFitResult:
    vertices: np.ndarray  # [T, V, 3]
    joints: np.ndarray  # [T, 22, 3]
    betas: np.ndarray  # [10]
    mean_joint_error_m: float


def default_smplh_model_dir() -> Path:
    raw = os.environ.get("SMPLH_MODEL_PATH", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    for candidate in (
        Path(os.environ.get("SMPL_MODEL_PATH", "")).expanduser(),
        Path.home() / "models" / "smplh",
        Path("/mnt/models/smplh"),
    ):
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "SMPL-H model not found. Set SMPLH_MODEL_PATH to a directory containing "
        "SMPLH_NEUTRAL.pkl (and male/female variants)."
    )


@lru_cache(maxsize=1)
def _load_smplh_model(model_dir: str, gender: str = "neutral"):
    import smplx

    path = Path(model_dir)
    return smplx.create(
        str(path),
        model_type="smplh",
        gender=str(gender),
        use_pca=False,
        num_pca_comps=12,
        flat_hand_mean=True,
        batch_size=1,
    )


def fit_smplh_to_hml_joints(
    joints: np.ndarray,
    *,
    model_dir: str | Path | None = None,
    gender: str = "neutral",
    num_iters: int = 120,
    lr: float = 0.02,
    device: str = "cpu",
) -> SmplhFitResult:
    """Optimize SMPL-H pose/shape/translation to match HML joint positions."""
    target = torch.tensor(np.asarray(joints, dtype=np.float32), device=device)
    if target.ndim != 3 or target.shape[1] != NUM_HML3D_JOINTS or target.shape[2] != 3:
        raise ValueError(f"Expected joints [T, 22, 3], got {tuple(target.shape)}")
    t_len = int(target.shape[0])
    if t_len < 1:
        raise ValueError("Empty joint sequence")

    model = _load_smplh_model(str(model_dir or default_smplh_model_dir()), gender=gender)
    model = model.to(device)
    model.eval()

    betas = torch.zeros((1, model.num_betas), device=device, requires_grad=True)
    global_orient = torch.zeros((t_len, 3), device=device, requires_grad=True)
    body_pose = torch.zeros((t_len, 63), device=device, requires_grad=True)
    transl = torch.zeros((t_len, 3), device=device, requires_grad=True)
    left_hand = torch.zeros((t_len, 45), device=device)
    right_hand = torch.zeros((t_len, 45), device=device)

    opt = torch.optim.Adam([betas, global_orient, body_pose, transl], lr=float(lr))
    for _ in range(int(num_iters)):
        opt.zero_grad()
        out = model(
            betas=betas.expand(t_len, -1),
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            left_hand_pose=left_hand,
            right_hand_pose=right_hand,
            return_verts=True,
        )
        pred_j = out.joints[:, :NUM_HML3D_JOINTS, :]
        loss = torch.mean((pred_j - target) ** 2)
        loss.backward()
        opt.step()

    with torch.no_grad():
        out = model(
            betas=betas.expand(t_len, -1),
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            left_hand_pose=left_hand,
            right_hand_pose=right_hand,
            return_verts=True,
        )
        pred_j = out.joints[:, :NUM_HML3D_JOINTS, :]
        err = float(torch.sqrt(torch.mean((pred_j - target) ** 2)).cpu().item())
        verts = out.vertices.detach().cpu().numpy().astype(np.float32)
        joints_out = pred_j.detach().cpu().numpy().astype(np.float32)
        betas_out = betas.detach().cpu().numpy().reshape(-1).astype(np.float32)

    return SmplhFitResult(
        vertices=verts,
        joints=joints_out,
        betas=betas_out,
        mean_joint_error_m=err,
    )
