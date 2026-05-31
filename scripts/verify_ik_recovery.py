#!/usr/bin/env python3
"""Verify direct Rajagopal IK pipeline completes on a sample motion."""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common.paths import default_humanml3d_root
from datasets.hml3d_joints import load_hml3d_joint_positions
from nimble.ik import fit_q
from nimble.physics import load_model


def main() -> None:
    motion_id = "000000"
    root = default_humanml3d_root()
    joints, _ = load_hml3d_joint_positions(root, motion_id, joint_source="new_joints")
    parsed = load_model()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        q, stats = fit_q(joints, parsed.skeleton)

    print(f"motion_id={motion_id}")
    print(f"frames={q.shape[1]} dofs={q.shape[0]}")
    print(f"mean_ik_error={stats.get('mean_ik_error', 0.0):.6e}")
    print(f"success_ratio={stats.get('success_ratio', 0.0):.4f}")
    print("pipeline_ok=True")


if __name__ == "__main__":
    main()
