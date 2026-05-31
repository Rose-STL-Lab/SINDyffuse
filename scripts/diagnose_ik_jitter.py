#!/usr/bin/env python3
"""Diagnose per-frame IK jitter on one HumanML3D motion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common.paths import default_humanml3d_root, nimble_b3d_dir
from datasets.nimble_dataset import read_q_segment
from nimble.ik import fill_invalid_pose_frames, fit_q, pose_is_invalid
from nimble.physics import load_model
from nimble.smoothing import apply_pose_smoothing_numpy, unwrap_pose_angles

MOTION_ID = "000000"
FPS = 20.0


def _dq_stats(q: np.ndarray) -> dict:
    """q: [dof, T]"""
    dq = np.diff(q, axis=1)
    norms = np.linalg.norm(dq, axis=0)
    return {
        "dq_mean": float(np.mean(norms)),
        "dq_median": float(np.median(norms)),
        "dq_max": float(np.max(norms)),
        "dq_p95": float(np.percentile(norms, 95)),
        "spike_frames": int(np.sum(norms > np.percentile(norms, 95) * 2.5 + 1e-9)),
    }


def _zero_frame_mask(q: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    return np.array([pose_is_invalid(q[:, t], atol=atol) for t in range(q.shape[1])])


def main() -> None:
    root = Path(default_humanml3d_root())
    joints = np.load(root / "new_joints" / f"{MOTION_ID}.npy").astype(np.float64)
    if joints.ndim == 3 and joints.shape[1] >= 22:
        joints = joints[:, :22, :]

    sk = load_model().skeleton
    print(f"Motion {MOTION_ID}: T={joints.shape[0]}, joints shape={joints.shape}")

  # Input joint motion smoothness
    jflat = joints.reshape(joints.shape[0], -1)
    jd = np.linalg.norm(np.diff(jflat, axis=0), axis=1)
    print("\n=== HumanML3D new_joints (input) ===")
    print(f"  frame-step XYZ norm: mean={jd.mean():.5f} med={np.median(jd):.5f} max={jd.max():.5f}")
    zero_j = sum(1 for t in range(joints.shape[0]) if np.allclose(joints[t], 0, atol=1e-8))
    print(f"  all-zero joint frames: {zero_j}")

    # Raw IK (no post smooth)
    q_raw, stats = fit_q(joints, sk)
    print("\n=== fit_q output (pre B3D smooth) ===")
    print(json.dumps({k: stats[k] for k in sorted(stats)}, indent=2))
    zmask = _zero_frame_mask(q_raw)
    print(f"  invalid/zero q frames: {int(zmask.sum())} / {q_raw.shape[1]}")
    if zmask.any():
        idx = np.where(zmask)[0]
        print(f"  zero frame indices (first 20): {idx[:20].tolist()}")

    dq = _dq_stats(q_raw)
    print(f"  dq norm: mean={dq['dq_mean']:.4f} med={dq['dq_median']:.4f} "
          f"p95={dq['dq_p95']:.4f} max={dq['dq_max']:.4f} spike_frames={dq['spike_frames']}")

    # Per-DOF spikes (rotation vs translation)
    dq_arr = np.diff(q_raw, axis=1)
    rot = dq_arr[0:3, :]
    trans = dq_arr[3:6, :]
    body = dq_arr[6:, :]
    print(f"  dq pelvis rot max: {np.max(np.linalg.norm(rot, axis=0)):.4f} rad")
    print(f"  dq pelvis trans max: {np.max(np.linalg.norm(trans, axis=0)):.4f} m")
    print(f"  dq body joints max: {np.max(np.linalg.norm(body, axis=0)):.4f} rad")

    # Identify top spike frames
    norms = np.linalg.norm(dq_arr, axis=0)
    top = np.argsort(norms)[-8:][::-1]
    print(f"  top |dq| frame indices (t→t+1): {[(int(i), float(norms[i])) for i in top]}")

    # Held-frame runs (consecutive identical q)
    same_as_prev = 0
    run_lengths = []
    run = 0
    for t in range(1, q_raw.shape[1]):
        if np.allclose(q_raw[:, t], q_raw[:, t - 1], atol=1e-8):
            same_as_prev += 1
            run += 1
        else:
            if run > 0:
                run_lengths.append(run)
            run = 0
    if run > 0:
        run_lengths.append(run)
    print(f"  frames identical to previous: {same_as_prev}")
    if run_lengths:
        print(f"  hold-run lengths (max): {max(run_lengths)}")

    # Post smooth like export
    q_t = np.ascontiguousarray(q_raw.T, dtype=np.float32)
    q_sm, _ = apply_pose_smoothing_numpy(q_t, fps=FPS, smooth_cutoff_hz=4.0)
    q_smooth = q_sm.T.astype(np.float64)
    dq2 = _dq_stats(q_smooth)
    print("\n=== after 4Hz pose smooth ===")
    print(f"  dq norm: mean={dq2['dq_mean']:.4f} max={dq2['dq_max']:.4f} "
          f"spike_frames={dq2['spike_frames']}")

    # Compare to cached B3D
    b3d = nimble_b3d_dir(root) / f"{MOTION_ID}.b3d"
    if b3d.is_file():
        q_b3d = read_q_segment(str(b3d)).T.astype(np.float64)
        if q_b3d.shape == q_raw.shape:
            diff = np.max(np.abs(q_b3d - q_raw))
            print(f"\n=== cached B3D vs fresh IK max abs diff: {diff:.6f}")
        z_b = _zero_frame_mask(q_b3d)
        print(f"  B3D invalid/zero frames: {int(z_b.sum())}")
        dq_b = _dq_stats(q_b3d)
        print(f"  B3D dq max: {dq_b['dq_max']:.4f}")

    # Angle unwrap diagnostic
    q_unwrap = unwrap_pose_angles(q_raw.T).T
    du = _dq_stats(q_unwrap)
    print("\n=== after unwrap only (no FIR) ===")
    print(f"  dq max: {du['dq_max']:.4f} (if << raw, 2pi wraps were issue)")


if __name__ == "__main__":
    main()
