#!/usr/bin/env python3
"""Visualize retargeted HumanML3D motion from the nimble_b3d cache.

Opens the Nimble web GUI (Three.js viewer) and plays generalized coordinates
from a per-motion ``.b3d`` file produced by ``preprocess_nimble.py``.

Examples (from the SINDyffuse repo root)::

  python scripts/visualize_nimble_b3d.py --motion_id 000000
  python scripts/visualize_nimble_b3d.py --b3d_path datasets/HumanML3D/nimble_b3d/000001.b3d

Then open http://localhost:8080 in a browser (opened automatically unless
``--no-browser`` is set). Press Ctrl+C in the terminal to stop.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import nimblephysics as nimble

from common.paths import nimble_b3d_dir, resolve_data_root
from datasets.nimble_dataset import read_q_segment
from datasets.splits import kinematics_pass_index
from nimble.physics import load_model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Play a nimble_b3d retargeted motion in the Nimble web GUI.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--motion_id",
        type=str,
        help="HumanML3D motion id (loads ``<data_root>/nimble_b3d/<id>.b3d``).",
    )
    src.add_argument(
        "--b3d_path",
        type=str,
        help="Path to a ``.b3d`` file (absolute or relative to repo root).",
    )
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="HumanML3D root (default: $HUMANML3D_ROOT or datasets/HumanML3D).",
    )
    p.add_argument("--trial", type=int, default=0, help="Trial index inside the B3D.")
    p.add_argument("--port", type=int, default=8080, help="HTTP port for NimbleGUI.")
    p.add_argument("--start_frame", type=int, default=None, help="First frame (inclusive).")
    p.add_argument("--num_frames", type=int, default=None, help="Number of frames to play.")
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (1.0 = real time from B3D timestep).",
    )
    p.add_argument(
        "--no_loop",
        action="store_true",
        help="Stop after one pass instead of looping.",
    )
    p.add_argument(
        "--no_browser",
        action="store_true",
        help="Do not open a browser tab automatically.",
    )
    return p.parse_args()


def _resolve_b3d_path(args: argparse.Namespace) -> Path:
    if args.b3d_path:
        path = Path(args.b3d_path).expanduser()
        if not path.is_absolute():
            path = (_REPO / path).resolve()
        return path
    root = resolve_data_root(args.data_root)
    path = nimble_b3d_dir(root) / f"{args.motion_id.strip()}.b3d"
    return path.resolve()


def _load_caption(hml_root: Path, motion_id: str) -> Optional[str]:
    text_path = hml_root / "texts" / f"{motion_id}.txt"
    if not text_path.is_file():
        return None
    lines = [ln.strip() for ln in text_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines[0].split("#")[0].strip() if lines else None


def _schedule_browser(url: str, *, delay_s: float = 0.8) -> None:
    def _open() -> None:
        time.sleep(delay_s)
        try:
            webbrowser.open(url, new=1, autoraise=True)
        except webbrowser.Error as exc:
            print(f"Could not open browser automatically: {exc}")
            print(f"Open manually: {url}")

    threading.Thread(target=_open, daemon=True).start()


def _animate(
    gui: nimble.NimbleGUI,
    skeleton: nimble.dynamics.Skeleton,
    poses: np.ndarray,
    dt: float,
    *,
    loop: bool,
    speed: float,
) -> None:
    sleep_s = max(dt / max(speed, 1e-6), 1e-4)
    while True:
        for t in range(poses.shape[0]):
            skeleton.setPositions(poses[t])
            gui.nativeAPI().renderSkeleton(skeleton)
            time.sleep(sleep_s)
        if not loop:
            break


def main() -> None:
    args = _parse_args()
    b3d_path = _resolve_b3d_path(args)
    if not b3d_path.is_file():
        raise FileNotFoundError(f"B3D not found: {b3d_path}")

    subj = nimble.biomechanics.SubjectOnDisk(str(b3d_path))
    trial = int(args.trial)
    kinematics_pass_index(subj, trial)  # validate the trial has a kinematics pass
    tlen = int(subj.getTrialLength(trial))
    if tlen < 1:
        raise RuntimeError(f"Trial {trial} is empty in {b3d_path}")

    seg_start = args.start_frame
    seg_end = None
    if args.num_frames is not None:
        start = 0 if seg_start is None else int(seg_start)
        seg_end = start + int(args.num_frames)

    poses = read_q_segment(
        str(b3d_path),
        trial=trial,
        seg_start=seg_start,
        seg_end=seg_end,
    )
    skeleton = load_model(with_geometry=True).skeleton
    n_dst = int(skeleton.getNumDofs())
    if poses.shape[1] != n_dst:
        raise RuntimeError(
            f"B3D q has {poses.shape[1]} DOFs but the bundled Rajagopal skeleton "
            f"has {n_dst}; the B3D was likely produced by a different model."
        )

    dt = float(subj.getTrialTimestep(trial))
    url = f"http://localhost:{int(args.port)}"

    if args.motion_id:
        caption = _load_caption(Path(resolve_data_root(args.data_root)), args.motion_id)
        if caption:
            print(f"Caption: {caption}")

    print(f"B3D: {b3d_path}")
    print(f"Frames: {poses.shape[0]}, DOFs: {poses.shape[1]}, dt: {dt:.4f}s, GUI: {url}")

    gui = nimble.NimbleGUI()
    gui.serve(int(args.port))
    skeleton.setPositions(poses[0])
    gui.nativeAPI().renderSkeleton(skeleton)

    if not args.no_browser:
        _schedule_browser(url)

    try:
        _animate(
            gui,
            skeleton,
            poses,
            dt,
            loop=not args.no_loop,
            speed=float(args.speed),
        )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        gui.blockWhileServing()


if __name__ == "__main__":
    main()
