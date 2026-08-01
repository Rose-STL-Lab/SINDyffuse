from __future__ import annotations
import os
from pathlib import Path
from typing import Literal, Optional
import numpy as np
import torch
from datasets.hml3d_motion import recover_from_ric
JointSource = Literal['auto', 'joints', 'new_joints']

def load_hml3d_joint_positions(hml_root: str | Path, motion_id: str, *, joint_source: JointSource='auto', joints_root: str | Path | None=None) -> tuple[np.ndarray, str]:
    root = Path(hml_root).expanduser().resolve()
    joints_dir = Path(joints_root).expanduser().resolve() if joints_root else root / 'joints'
    order: tuple[str, ...]
    if joint_source == 'joints':
        order = ('joints',)
    elif joint_source == 'new_joints':
        order = ('new_joints',)
    else:
        order = ('joints', 'new_joints')
    for label in order:
        if label == 'joints':
            path = joints_dir / f'{motion_id}.npy'
        else:
            path = root / 'new_joints' / f'{motion_id}.npy'
        if not path.is_file():
            continue
        raw = np.load(path).astype(np.float64)
        if raw.ndim == 3 and raw.shape[1] >= 22:
            return (raw[:, :22, :], label)
    vecs_p = root / 'new_joint_vecs' / f'{motion_id}.npy'
    if vecs_p.is_file():
        vec = np.load(vecs_p).astype(np.float32)
        if vec.ndim == 2 and vec.shape[1] >= 263:
            mean_p, std_p = (root / 'Mean.npy', root / 'Std.npy')
            if mean_p.is_file() and std_p.is_file():
                denorm = vec * np.load(std_p).astype(np.float32) + np.load(mean_p).astype(np.float32)
                with torch.no_grad():
                    j = recover_from_ric(torch.from_numpy(denorm).unsqueeze(0), 22)
                out = j.squeeze(0).numpy().astype(np.float64)
                if out.ndim == 3 and out.shape[1] >= 22:
                    return (out[:, :22, :], 'new_joint_vecs')
    raise FileNotFoundError(f'Could not load 22×3 joints for {motion_id!r} under {root} (joint_source={joint_source!r}, joints_root={joints_dir})')

def default_joints_root(hml_root: str | Path) -> Optional[Path]:
    explicit = os.environ.get('HUMANML3D_JOINTS_ROOT', '').strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidate = Path(hml_root).expanduser().resolve() / 'joints'
    return candidate if candidate.is_dir() else None