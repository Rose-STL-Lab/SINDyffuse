#!/usr/bin/env python3
"""Per-frame + per-joint IK error analysis for one HumanML3D motion.

Run from the SINDyffuse repo root::

    python scripts/analyze_ik_per_frame.py
    python scripts/analyze_ik_per_frame.py --motion_id 000123
    python scripts/analyze_ik_per_frame.py --out_json /tmp/ik_frame_audit.json

The goal: take a HumanML3D motion, refit Rajagopal ``q``, and emit per-frame
metrics (IK fit error, dq-norm, joint accel) so we can decide whether visible
visualization jitter is driven by bad IK on specific frames or by genuinely
jittery input poses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common.paths import default_humanml3d_root
from datasets.hml3d_joints import load_hml3d_joint_positions
from nimble.ik import _get_joint_ik_cache, fit_q
from nimble.physics import load_model
from nimble.skeleton_registry import get_spec

# HumanML3D ``new_joints`` order (same labels as analyze_ik_by_joint.py).
HML_NAMES: Tuple[str, ...] = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)


def _per_frame_joint_errors(
    skeleton, cache, poses_q: np.ndarray, hml_positions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Recompute per-frame per-joint errors via FK on the fitted ``q``.

    Returns:
        per_frame_loss[T]           sum-of-squared errors across IK targets (same metric as fit_q)
        per_joint_err[T, K]         Euclidean distance per IK-mapped joint
    """
    t_frames = int(poses_q.shape[1])
    k_targets = len(cache.joints)
    per_frame_loss = np.zeros(t_frames, dtype=np.float64)
    per_joint_err = np.zeros((t_frames, k_targets), dtype=np.float64)
    for t in range(t_frames):
        skeleton.setPositions(poses_q[:, t])
        skeleton.computeForwardKinematics()
        flat = np.asarray(
            skeleton.getJointWorldPositions(list(cache.joints)), dtype=np.float64
        ).reshape(-1, 3)
        targets = np.array(
            [hml_positions[t, int(j)] for j in cache.hml_indices], dtype=np.float64
        )
        diff = flat - targets
        per_joint_err[t] = np.linalg.norm(diff, axis=1)
        per_frame_loss[t] = float(np.sum(diff * diff))
    return per_frame_loss, per_joint_err


def _temporal_metrics(
    poses_q: np.ndarray, hml_positions: np.ndarray
) -> dict:
    """``q`` and joint-position temporal jitter metrics."""
    # poses_q [ndof, T]; we'd like per-frame norm of dq.
    dq = np.diff(poses_q, axis=1)  # [ndof, T-1]
    dq_norm = np.linalg.norm(dq, axis=0)  # [T-1]
    ddq = np.diff(dq, axis=1)  # [ndof, T-2]
    ddq_norm = np.linalg.norm(ddq, axis=0)  # [T-2]

    # Joint XYZ jitter on the input HumanML3D positions (independent of IK).
    jp = hml_positions.reshape(hml_positions.shape[0], -1)  # [T, 66]
    djp = np.diff(jp, axis=0)
    djp_norm = np.linalg.norm(djp, axis=1)  # [T-1]
    ddjp = np.diff(djp, axis=0)
    ddjp_norm = np.linalg.norm(ddjp, axis=1)  # [T-2]
    return {
        "dq_norm": dq_norm,
        "ddq_norm": ddq_norm,
        "djp_norm": djp_norm,
        "ddjp_norm": ddjp_norm,
    }


def _outlier_frames(values: np.ndarray, *, k: float = 4.0) -> List[int]:
    """Indices where ``values`` exceed median + k * MAD."""
    finite = np.isfinite(values)
    if not finite.any():
        return []
    med = float(np.median(values[finite]))
    mad = float(np.median(np.abs(values[finite] - med))) or 1e-12
    thr = med + k * mad
    return [int(i) for i in np.where(values > thr)[0]]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    a2, b2 = a[mask], b[mask]
    if a2.std() < 1e-12 or b2.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a2, b2)[0, 1])


def _summarize(arr: np.ndarray) -> dict:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {k: float("nan") for k in ("mean", "median", "std", "p95", "max")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion_id", type=str, default="000000")
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of worst frames / joints to surface in the text report.",
    )
    parser.add_argument(
        "--mad_k",
        type=float,
        default=4.0,
        help="Outlier threshold (median + k * MAD) for flagging spike frames.",
    )
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()

    spec = get_spec()
    parsed = load_model()
    sk = parsed.skeleton

    root = Path(default_humanml3d_root())
    poses_xyz, _ = load_hml3d_joint_positions(root, args.motion_id, joint_source="new_joints")
    t_frames = int(poses_xyz.shape[0])

    print(
        f"Motion {args.motion_id}  skeleton={spec.name}  ndof={sk.getNumDofs()}  "
        f"frames={t_frames}  IK-targets={len(spec.ik_mapping)}",
        flush=True,
    )

    poses_q, stats = fit_q(poses_xyz, sk, ik_mapping=spec.ik_mapping)
    cache = _get_joint_ik_cache(sk, ik_mapping=spec.ik_mapping)

    per_frame_loss, per_joint_err = _per_frame_joint_errors(sk, cache, poses_q, poses_xyz)
    rms_per_frame = np.sqrt(per_frame_loss / max(len(cache.joints), 1))  # avg m per joint per frame
    temporal = _temporal_metrics(poses_q, poses_xyz)
    dq_norm = temporal["dq_norm"]
    ddq_norm = temporal["ddq_norm"]
    djp_norm = temporal["djp_norm"]
    ddjp_norm = temporal["ddjp_norm"]

    # ---- aggregate summary ----
    print("\n=== Aggregate (across all frames) ===")
    print(
        f"  IK loss   (sum-sq err  m^2): {_summarize(per_frame_loss)}"
    )
    print(
        f"  IK RMS    (per joint, m)   : {_summarize(rms_per_frame)}"
    )
    print(
        f"  |dq|       (rad+m per frm) : {_summarize(dq_norm)}"
    )
    print(
        f"  |ddq|      (jerk in q)     : {_summarize(ddq_norm)}"
    )
    print(
        f"  |d hml_xyz|  (input jitter): {_summarize(djp_norm)}"
    )
    print(
        f"  |dd hml_xyz| (input jerk)  : {_summarize(ddjp_norm)}"
    )

    # ---- top-N worst frames by IK error ----
    top_err = np.argsort(-per_frame_loss)[: args.top_n]
    print(f"\n=== Top {args.top_n} worst-IK frames ===")
    print(f"  {'frame':>6}  {'loss(m^2)':>11}  {'rms(m)':>9}  {'|dq|':>9}  {'|ddq|':>9}  worst joints")
    for f in top_err:
        worst_k = np.argsort(-per_joint_err[f])[:3]
        worst_names = [
            f"{HML_NAMES[cache.hml_indices[k]]}={per_joint_err[f, k]:.3f}m" for k in worst_k
        ]
        dqv = float(dq_norm[f]) if 0 <= f < len(dq_norm) else float("nan")
        ddqv = float(ddq_norm[f]) if 0 <= f < len(ddq_norm) else float("nan")
        print(
            f"  {int(f):>6}  {per_frame_loss[f]:>11.6f}  {rms_per_frame[f]:>9.4f}  "
            f"{dqv:>9.4f}  {ddqv:>9.4f}  {', '.join(worst_names)}"
        )

    # ---- top-N jitter frames by |dq| ----
    top_dq = np.argsort(-dq_norm)[: args.top_n]
    print(f"\n=== Top {args.top_n} worst-jitter frames (by |dq|, t -> t+1) ===")
    print(f"  {'frame':>6}  {'|dq|':>9}  {'|ddq|':>9}  {'loss@t':>10}  {'loss@t+1':>10}  {'|d hml_xyz|':>11}")
    for f in top_dq:
        lt = per_frame_loss[f] if f < t_frames else float("nan")
        lt1 = per_frame_loss[f + 1] if (f + 1) < t_frames else float("nan")
        ddqv = float(ddq_norm[f]) if 0 <= f < len(ddq_norm) else float("nan")
        djp = float(djp_norm[f]) if 0 <= f < len(djp_norm) else float("nan")
        print(
            f"  {int(f):>6}  {dq_norm[f]:>9.4f}  {ddqv:>9.4f}  {lt:>10.6f}  {lt1:>10.6f}  {djp:>11.4f}"
        )

    # ---- correlations ----
    # Align lengths: loss is [T], dq_norm is [T-1].  Pair (t, t->t+1) by using loss at t+1
    # (the "post-jump" frame) and dq at t.
    if len(dq_norm) >= 3:
        loss_post = per_frame_loss[1:]  # [T-1]
        loss_pre = per_frame_loss[:-1]  # [T-1]
        c_jitter_post_err = _corr(dq_norm, loss_post)
        c_jitter_pre_err = _corr(dq_norm, loss_pre)
        c_jitter_input = _corr(dq_norm, djp_norm)
        print("\n=== Correlations (Pearson r) ===")
        print(f"  |dq[t]| vs IK loss[t+1] : {c_jitter_post_err:+.3f}  (does a q-jump land on a bad fit?)")
        print(f"  |dq[t]| vs IK loss[t]   : {c_jitter_pre_err:+.3f}  (does a q-jump leave a bad fit?)")
        print(f"  |dq[t]| vs |d hml_xyz[t]|: {c_jitter_input:+.3f}  (does q-jitter track input jitter?)")

    # ---- spike frames (outlier-flagged) ----
    spike_dq = _outlier_frames(dq_norm, k=args.mad_k)
    spike_loss = _outlier_frames(per_frame_loss, k=args.mad_k)
    spike_ddq = _outlier_frames(ddq_norm, k=args.mad_k)
    print(f"\n=== Outliers (median + {args.mad_k:.1f} * MAD) ===")
    print(f"  IK-loss spike frames ({len(spike_loss)}): {spike_loss[:30]}{'...' if len(spike_loss) > 30 else ''}")
    print(f"  |dq|  spike frames ({len(spike_dq)}): {spike_dq[:30]}{'...' if len(spike_dq) > 30 else ''}")
    print(f"  |ddq| spike frames ({len(spike_ddq)}): {spike_ddq[:30]}{'...' if len(spike_ddq) > 30 else ''}")
    overlap_dq_loss = sorted(set(spike_dq) & set(spike_loss))
    print(
        f"  frames spiking in BOTH |dq| and IK-loss: {len(overlap_dq_loss)}  -> {overlap_dq_loss[:30]}"
    )

    # ---- per-joint summary across motion ----
    print("\n=== Per-IK-target mean error across this motion ===")
    joint_means = per_joint_err.mean(axis=0)
    joint_maxs = per_joint_err.max(axis=0)
    order = np.argsort(-joint_means)
    print(f"  {'joint':25s}  {'mean(m)':>9}  {'max(m)':>9}  {'arg max @ frame':>16}")
    for k in order:
        hml_name = HML_NAMES[cache.hml_indices[k]]
        skel_name = cache.joint_names[k]
        argmax_t = int(np.argmax(per_joint_err[:, k]))
        print(
            f"  {hml_name+' ['+skel_name+']':25s}  {joint_means[k]:>9.4f}  "
            f"{joint_maxs[k]:>9.4f}  {argmax_t:>16}"
        )

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "motion_id": args.motion_id,
            "skeleton": spec.name,
            "ndof": int(sk.getNumDofs()),
            "frames": int(t_frames),
            "ik_target_joints": list(cache.joint_names),
            "ik_target_hml_idx": list(cache.hml_indices),
            "per_frame_loss_m2": per_frame_loss.tolist(),
            "per_frame_rms_m": rms_per_frame.tolist(),
            "per_joint_err_m": per_joint_err.tolist(),
            "dq_norm": dq_norm.tolist(),
            "ddq_norm": ddq_norm.tolist(),
            "djp_norm_input": djp_norm.tolist(),
            "ddjp_norm_input": ddjp_norm.tolist(),
            "stats": stats,
            "spike_frames": {
                "ik_loss": spike_loss,
                "dq": spike_dq,
                "ddq": spike_ddq,
                "dq_and_ik_loss": overlap_dq_loss,
            },
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
