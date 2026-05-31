#!/usr/bin/env python3
"""Compute OpenSim muscle activations from a Nimble B3D ``q`` trajectory (smoke test / offline)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common.paths import default_humanml3d_root, nimble_b3d_dir, resolve_data_root
from datasets.nimble_dataset import read_q_segment
from surrogate.opensim_activation import (
    MuscleActivationConfig,
    activation_stats,
    compute_muscle_activation,
    muscle_names,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OpenSim Static Optimization on a retargeted B3D motion.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="HumanML3D root (default: datasets/HumanML3D).",
    )
    parser.add_argument(
        "--motion_id",
        type=str,
        default="000000",
        help="Motion id (loads <data_root>/nimble_b3d/<id>.b3d).",
    )
    parser.add_argument(
        "--b3d_path",
        type=str,
        default=None,
        help="Explicit .b3d path (overrides --motion_id).",
    )
    parser.add_argument("--start_frame", type=int, default=0, help="Segment start frame.")
    parser.add_argument(
        "--num_frames",
        type=int,
        default=64,
        help="Number of frames to read (default 64).",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Frame rate.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .npz path (default: <b3d_stem>_activations.npz).",
    )
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep OpenSim working directory for debugging.",
    )
    args = parser.parse_args()

    root = resolve_data_root(args.data_root or default_humanml3d_root())
    if args.b3d_path:
        b3d_path = Path(args.b3d_path).expanduser().resolve()
    else:
        b3d_path = nimble_b3d_dir(root) / f"{args.motion_id.strip()}.b3d"
    if not b3d_path.is_file():
        print(f"ERROR: B3D not found: {b3d_path}", flush=True)
        sys.exit(1)

    st = int(args.start_frame)
    n = int(args.num_frames)
    q = read_q_segment(str(b3d_path), seg_start=st, seg_end=st + n).astype(np.float64)
    if q.shape[0] < 2:
        print(f"ERROR: Need at least 2 frames, got {q.shape[0]}", flush=True)
        sys.exit(1)

    cfg = MuscleActivationConfig(fps=float(args.fps), keep_temp=bool(args.keep_temp))
    result = compute_muscle_activation(q, cfg=cfg)
    stats = activation_stats(result.activations)

    out_path = Path(args.output) if args.output else b3d_path.with_suffix("").with_name(
        b3d_path.stem + "_activations.npz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        activations=result.activations,
        muscle_names=np.asarray(result.muscle_names, dtype=object),
        success_mask=result.success_mask,
        mean_activation=stats["mean_activation"],
        max_activation=stats["max_activation"],
        activation_smoothness=stats["activation_smoothness"],
        q=q.astype(np.float32),
    )

    summary = {
        "b3d_path": str(b3d_path),
        "output": str(out_path),
        "num_frames": int(result.num_frames),
        "num_muscles": int(result.num_muscles),
        "success_fraction": float(result.success_mask.mean()),
        **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in result.metadata.items()},
    }
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {out_path} activations {result.activations.shape}", flush=True)
    print(f"Summary {summary_path}", flush=True)
    print(f"Muscles ({len(muscle_names())}): first={result.muscle_names[0]}", flush=True)


if __name__ == "__main__":
    main()
