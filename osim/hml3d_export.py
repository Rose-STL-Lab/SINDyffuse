"""
HumanML3D → NumPy biomechanics export (OpenSim-first).

Writes compressed NumPy archives (`.npz`) under `features_v1/`, via `osim.pipeline`.
Includes Rajagopal/HumanML3D naming maps for downstream OpenSim IK setup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from osim.npz_io import save_npz
from osim.paths import DEFAULT_HML3D_JOINTS_DIR, DEFAULT_MODEL_PATH, DEFAULT_NPZ_EXPORT_DIR
from osim.pipeline import from_hml3d


def get_hml3d_to_rajagopal_mapping() -> Dict[int, str]:
    """HumanML3D 22-joint layout mapped to Rajagopal-style body names (reference only)."""
    return {
        0: "pelvis",
        1: "femur_l",
        2: "femur_r",
        3: "torso",
        4: "tibia_l",
        5: "tibia_r",
        6: "torso",
        7: "talus_l",
        8: "talus_r",
        9: "torso",
        10: "toes_l",
        11: "toes_r",
        12: "torso",
        13: "humerus_l",
        14: "humerus_r",
        15: "torso",
        16: "ulna_l",
        17: "ulna_r",
        18: "radius_l",
        19: "radius_r",
        20: "radius_l",
        21: "radius_r",
    }


def get_rajagopal_nodes_for_ik() -> List[str]:
    """Ordered body-node names commonly used when fitting Rajagopal to sparse targets."""
    return [
        "pelvis",
        "femur_l",
        "tibia_l",
        "talus_l",
        "toes_l",
        "femur_r",
        "tibia_r",
        "talus_r",
        "toes_r",
        "torso",
        "humerus_l",
        "ulna_l",
        "radius_l",
        "humerus_r",
        "ulna_r",
        "radius_r",
    ]


def convert_file(
    input_npy: str,
    output_dir: str,
    fps: int = 20,
    sampling_frequency: Optional[float] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    *,
    smooth_poses: bool = False,
    smooth_cutoff_hz: float = 6.0,
    smooth_butterworth_order: int = 2,
) -> Dict[str, Any]:
    """Convert one HumanML3D `[T,22,3]` motion `.npy` to `features_v1/*.npz` under *output_dir*."""
    stem = Path(input_npy).stem
    result = from_hml3d(
        input_path=input_npy,
        fps=int(fps),
        sampling_frequency=sampling_frequency,
        model_path=model_path,
        smooth_poses=smooth_poses,
        smooth_cutoff_hz=smooth_cutoff_hz,
        smooth_butterworth_order=smooth_butterworth_order,
    )
    saved = save_npz(result["memory_result"], output_dir, file_stem=stem)
    return {"input_npy": input_npy, "saved": saved}


def convert_directory(
    input_dir: str,
    output_dir: str,
    fps: int = 20,
    sampling_frequency: Optional[float] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    *,
    smooth_poses: bool = False,
    smooth_cutoff_hz: float = 6.0,
    smooth_butterworth_order: int = 2,
) -> Dict[str, Any]:
    """Batch-convert `.npy` motions to `features_v1/*.npz` (same layout as the simulator CLI)."""
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(in_dir.rglob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found under {in_dir}")

    converted = 0
    skipped = 0
    trial_rows: List[Dict[str, Any]] = []

    for npy_path in npy_files:
        try:
            arr = np.load(npy_path).astype(np.float32)
            if arr.ndim == 2 and arr.shape == (22, 3):
                ok = True
            elif arr.ndim == 3 and arr.shape[1:] == (22, 3):
                ok = True
            else:
                ok = False
            if not ok:
                skipped += 1
                continue

            trial_rows.append(
                convert_file(
                    input_npy=str(npy_path),
                    output_dir=str(out_dir),
                    fps=int(fps),
                    sampling_frequency=sampling_frequency,
                    model_path=model_path,
                    smooth_poses=smooth_poses,
                    smooth_cutoff_hz=smooth_cutoff_hz,
                    smooth_butterworth_order=smooth_butterworth_order,
                )
            )
            converted += 1
        except Exception:
            skipped += 1

    summary = {
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "engine": "opensim_numpy_npz",
        "converted_trials": float(converted),
        "skipped_trials": float(skipped),
        "model_path": model_path,
    }
    summary_path = out_dir / "conversion_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "trials": trial_rows}, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HumanML3D motions to NumPy biomechanics archives (.npz)"
    )
    parser.add_argument("--input_dir", default=DEFAULT_HML3D_JOINTS_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_NPZ_EXPORT_DIR)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--sampling_frequency", type=float, default=None)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--smooth_poses", type=int, default=0)
    parser.add_argument("--smooth_cutoff_hz", type=float, default=6.0)
    parser.add_argument("--smooth_butterworth_order", type=int, default=2)
    args = parser.parse_args()

    summary = convert_directory(
        input_dir=str(args.input_dir),
        output_dir=str(args.output_dir),
        fps=int(args.fps),
        sampling_frequency=args.sampling_frequency,
        model_path=str(args.model_path),
        smooth_poses=bool(int(args.smooth_poses)),
        smooth_cutoff_hz=float(args.smooth_cutoff_hz),
        smooth_butterworth_order=int(args.smooth_butterworth_order),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
