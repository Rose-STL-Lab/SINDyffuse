"""CLI: HumanML3D → NPZ features + manifest."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import opensim  # type: ignore[import-untyped]

from osim.manifest import inspect_contact_model_config, sha256, trial_features_signature, write_run_manifest
from osim.npz_io import save_npz
from common.paths import DEFAULT_MODEL_PATH
from osim.pipeline import from_hml3d, from_hml3d_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute biomechanics from HumanML3D (features + mandatory MocoTrack)")
    parser.add_argument("--input_path", required=True, help="Single .npy file or directory")
    parser.add_argument("--output_dir", required=True, help="Output root directory")
    parser.add_argument("--input_type", choices=["auto", "hml3d_npy"], default="auto")
    parser.add_argument("--processing_pass", type=int, default=0)
    parser.add_argument("--include_sensor_data", type=int, default=1)
    parser.add_argument("--include_poses", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--sampling_frequency", type=float, default=None)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--smooth_poses",
        type=int,
        default=0,
        help="1: low-pass Cartesian joint paths before kinematic derivatives (reduces jitter; default off for parity).",
    )
    parser.add_argument(
        "--smooth_cutoff_hz",
        type=float,
        default=6.0,
        help="Low-pass cutoff in Hz before derivatives (SciPy SOS Butterworth; clamped < Nyquist).",
    )
    parser.add_argument("--smooth_butterworth_order", type=int, default=2, help="Butterworth filter order.")
    args = parser.parse_args()

    in_path = Path(args.input_path)
    out_dir = Path(args.output_dir)
    include_sensor = bool(int(args.include_sensor_data))
    include_poses = bool(int(args.include_poses))
    model_path = str(args.model_path)
    model_exists = Path(model_path).exists()

    smooth_poses_flag = bool(int(args.smooth_poses))
    smooth_kwargs = dict(
        smooth_poses=smooth_poses_flag,
        smooth_cutoff_hz=float(args.smooth_cutoff_hz),
        smooth_butterworth_order=int(args.smooth_butterworth_order),
    )

    if in_path.is_dir():
        mem_result = from_hml3d_dir(
            input_dir=str(in_path),
            processing_pass=int(args.processing_pass),
            include_sensor_data=include_sensor,
            include_poses=include_poses,
            fps=int(args.fps),
            sampling_frequency=args.sampling_frequency,
            model_path=model_path,
            **smooth_kwargs,
        )
        saved_rows = []
        for row in mem_result["rows"]:
            stem = Path(row["input_npy"]).stem
            saved_rows.append(save_npz(row["memory_result"], str(out_dir), file_stem=stem))
        pose_smooth_meta_snapshot = (
            mem_result["rows"][0]["memory_result"].get("pose_smoothing") if mem_result["rows"] else {}
        )
        payload = {
            "mode": "hml3d_dir",
            "engine": "opensim_moco",
            "input_path": str(in_path),
            "result": {"input_dir": str(in_path), "num_files": len(saved_rows), "rows": saved_rows},
        }
    else:
        mem = from_hml3d(
            input_path=str(in_path),
            processing_pass=int(args.processing_pass),
            include_sensor_data=include_sensor,
            include_poses=include_poses,
            fps=int(args.fps),
            sampling_frequency=args.sampling_frequency,
            model_path=model_path,
            **smooth_kwargs,
        )
        pose_smooth_meta_snapshot = mem["memory_result"].get("pose_smoothing", {})
        result_obj: Dict[str, Any] = {"in_memory_signature": trial_features_signature(mem["memory_result"])}
        result_obj["saved_npz"] = save_npz(mem["memory_result"], str(out_dir), file_stem=Path(in_path).stem)
        payload = {"mode": "hml3d_file", "engine": "opensim_moco", "input_path": str(in_path), "result": result_obj}

    payload["config"] = {
        "input_type": "hml3d_npy",
        "save_format": "npz_numpy_archive",
        "processing_pass": int(args.processing_pass),
        "include_sensor_data": include_sensor,
        "include_poses": include_poses,
        "fps": int(args.fps),
        "sampling_frequency": args.sampling_frequency,
        "model_path": model_path,
        "model_exists": bool(model_exists),
        "model_sha256": sha256(Path(model_path)) if model_exists else None,
        "contact_model": inspect_contact_model_config(model_path),
        "opensim_version": str(opensim.GetVersionAndDate()),
        "python": os.sys.version,
        "pose_smoothing": pose_smooth_meta_snapshot,
        "moco_tracking": True,
    }
    write_run_manifest(out_dir, payload)
    print(f"Done. Manifest: {out_dir / 'compute_biomechanics_manifest.json'}")


if __name__ == "__main__":
    main()
