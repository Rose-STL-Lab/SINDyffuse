"""Export Rajagopal keypoint indices for differentiable FK (run once or on first import)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from nimble.physics import load_model
from nimble.rajagopal_kin import KEYPOINT_JOINT_NAMES, foot_body_indices

_EXPORT_PATH = Path(__file__).resolve().parent / "rajagopal_kinematics.pt"
MAPPING_VERSION = 4


def build_kinematics_record() -> Dict[str, Any]:
    sk = load_model().skeleton
    ndof = int(sk.getNumDofs())
    foot_l, foot_r = foot_body_indices(sk)
    keypoint_joint_indices = np.asarray(
        [int(sk.getJoint(name).getJointIndexInSkeleton()) for name in KEYPOINT_JOINT_NAMES],
        dtype=np.int64,
    )
    q_lo = np.asarray(sk.getPositionLowerLimits(), dtype=np.float64).reshape(-1)
    q_hi = np.asarray(sk.getPositionUpperLimits(), dtype=np.float64).reshape(-1)
    return {
        "mapping_version": MAPPING_VERSION,
        "ndof": ndof,
        "keypoint_joint_indices": keypoint_joint_indices,
        "keypoint_joint_names": list(KEYPOINT_JOINT_NAMES),
        "foot_left_body": int(foot_l),
        "foot_right_body": int(foot_r),
        "q_lo": q_lo.astype(np.float32),
        "q_hi": q_hi.astype(np.float32),
    }


def export_kinematics(path: Path | None = None) -> Path:
    out = path or _EXPORT_PATH
    record = build_kinematics_record()
    torch.save(record, str(out))
    return out


def load_kinematics(path: Path | None = None) -> Dict[str, Any]:
    p = path or _EXPORT_PATH
    if not p.is_file():
        export_kinematics(p)
    else:
        try:
            record = torch.load(str(p), map_location="cpu", weights_only=False)
        except TypeError:
            record = torch.load(str(p), map_location="cpu")
        if not isinstance(record, dict) or record.get("mapping_version") != MAPPING_VERSION:
            export_kinematics(p)
    try:
        record = torch.load(str(p), map_location="cpu", weights_only=False)
    except TypeError:
        record = torch.load(str(p), map_location="cpu")
    if isinstance(record, dict):
        return record
    raise RuntimeError(f"Unexpected kinematics export at {p}")


if __name__ == "__main__":
    p = export_kinematics()
    rec = load_kinematics(p)
    print(
        f"Wrote {p} ndof={rec['ndof']} "
        f"keypoints={len(rec['keypoint_joint_indices'])} "
        f"feet=({rec['foot_left_body']}, {rec['foot_right_body']})"
    )
