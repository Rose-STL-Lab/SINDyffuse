#!/usr/bin/env python3
"""Visualize a Nimble B3D motion as a Rajagopal mesh skeleton PDF figure."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.paths import humanml3d_text_dir, nimble_b3d_dir, resolve_data_root, resolve_repo_path, results_dir
from datasets.nimble_dataset import read_q_segment
from nimble.skeleton_registry import load_skeleton
from nimble.viz import (
    add_figure_caption,
    compute_height_offset,
    draw_rajagopal_montage,
    figure_content_bbox,
    fit_figure_to_projected_content,
    rajagopal_mesh_specs,
)

# --- user config (editable) ---
MOTION_ID: str | None = None
DATA_ROOT: str | None = None
OUTPUT_PDF: str | None = None
NUM_POSES: int = 8
RANDOM_SEED: int | None = None
CAPTION_FONT_SIZE: float = 8.0
CAPTION_GAP: float = 0.006  # axes-fraction gap above montage (lower = tighter)
VIEW: str = "default"  # "default" (oblique) or "flat" (side elevation)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Rajagopal mesh montage PDF from a Nimble B3D motion.",
    )
    parser.add_argument(
        "--motion_id",
        default=MOTION_ID,
        help="HumanML3D motion id (default: random .b3d from nimble_b3d/)",
    )
    parser.add_argument(
        "--data_root",
        default=DATA_ROOT,
        help="Dataset root containing nimble_b3d/ (default: datasets/HumanML3D)",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PDF,
        help="Output PDF path (default: results/viz/{motion_id}.pdf)",
    )
    parser.add_argument(
        "--num_poses",
        type=int,
        default=NUM_POSES,
        help="Number of evenly spaced poses in the montage",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed when choosing a motion id",
    )
    parser.add_argument(
        "--view",
        default=VIEW,
        choices=("default", "flat"),
        help='Camera preset: "default" oblique top-down, "flat" side view for feet/floor',
    )
    return parser.parse_args()


def _list_b3d_ids(b3d_dir: Path) -> list[str]:
    if not b3d_dir.is_dir():
        raise FileNotFoundError(f"Missing B3D cache directory: {b3d_dir}")
    ids = sorted(p.stem for p in b3d_dir.glob("*.b3d"))
    if not ids:
        raise FileNotFoundError(f"No .b3d files found under {b3d_dir}")
    return ids


def _choose_motion_id(b3d_dir: Path, motion_id: str | None, seed: int | None) -> str:
    available = _list_b3d_ids(b3d_dir)
    if motion_id is not None and str(motion_id).strip():
        chosen = str(motion_id).strip()
        b3d_path = b3d_dir / f"{chosen}.b3d"
        if not b3d_path.is_file():
            raise FileNotFoundError(f"Missing B3D file: {b3d_path}")
        return chosen

    rng = random.Random(seed)
    return rng.choice(available)


def _load_caption(data_root: Path, motion_id: str) -> str | None:
    text_path = humanml3d_text_dir(data_root) / f"{motion_id}.txt"
    if not text_path.is_file():
        return None
    first_line = text_path.read_text(encoding="utf-8").splitlines()[0].strip()
    if not first_line:
        return None
    return first_line.split("#", 1)[0].strip()


def _frame_indices(num_frames: int, num_poses: int) -> list[int]:
    if num_frames < 1:
        raise ValueError("Motion has no frames")
    n = max(1, int(num_poses))
    if n == 1:
        return [0]
    if num_frames == 1:
        return [0] * n
    return [int(round(i * (num_frames - 1) / (n - 1))) for i in range(n)]


def _default_output_path(motion_id: str, view: str) -> Path:
    return _apply_view_suffix(results_dir() / "viz" / f"{motion_id}.pdf", view)


def _apply_view_suffix(path: Path, view: str) -> Path:
    view_key = str(view).strip().lower()
    if view_key == "default":
        return path
    suffix = f"_{view_key}"
    if path.stem.endswith(suffix):
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _resolve_output_path(motion_id: str, output: str | None, view: str) -> Path:
    if output is None or str(output).strip() == "":
        return _default_output_path(motion_id, view)
    return _apply_view_suffix(resolve_repo_path(str(output)), view)


def render_b3d_motion_pdf(
    *,
    motion_id: str,
    data_root: str | Path,
    output_pdf: str | Path,
    num_poses: int = NUM_POSES,
    view: str = VIEW,
) -> Path:
    data_root = Path(resolve_data_root(str(data_root)))
    b3d_dir = nimble_b3d_dir(data_root)
    b3d_path = b3d_dir / f"{motion_id}.b3d"
    if not b3d_path.is_file():
        raise FileNotFoundError(f"Missing B3D file: {b3d_path}")

    q = read_q_segment(str(b3d_path))
    num_frames = int(q.shape[0])
    frame_ids = _frame_indices(num_frames, num_poses)

    skeleton = load_skeleton("rajagopal")[0].skeleton
    mesh_specs = rajagopal_mesh_specs(skeleton)
    if not mesh_specs:
        raise RuntimeError("Rajagopal skeleton has no mesh shape nodes to render")

    selected_q = [q[frame_idx] for frame_idx in frame_ids]
    height_offset = compute_height_offset(skeleton, selected_q, mesh_specs)

    n_poses = len(selected_q)
    fig_width = max(8.0, 1.6 + 1.4 * n_poses)
    fig = plt.figure(figsize=(fig_width, 3.0), dpi=150)
    caption = _load_caption(data_root, motion_id)

    ax = fig.add_subplot(1, 1, 1, projection="3d")
    mesh_mins, mesh_maxs = draw_rajagopal_montage(
        ax,
        skeleton,
        selected_q,
        mesh_specs,
        height_offset=height_offset,
        view=view,
    )
    ax.set_axis_off()

    caption_lines = 0
    if caption:
        from textwrap import wrap

        wrap_cols = max(28, int(fig_width * 0.96 * 7))
        caption_lines = len(wrap(caption, wrap_cols))

    top_margin = 0.03 + caption_lines * (CAPTION_FONT_SIZE / 72.0) * 1.15
    fit_figure_to_projected_content(
        fig,
        ax,
        mesh_mins,
        mesh_maxs,
        top_margin=top_margin,
    )

    caption_artist = None
    if caption:
        caption_artist = add_figure_caption(
            fig,
            ax,
            caption,
            fontsize=CAPTION_FONT_SIZE,
            gap=CAPTION_GAP,
        )

    out_path = Path(output_pdf).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop_bbox = figure_content_bbox(
        fig,
        ax,
        mesh_mins,
        mesh_maxs,
        extra_artists=[caption_artist] if caption_artist is not None else (),
    )
    fig.savefig(out_path, format="pdf", bbox_inches=crop_bbox, pad_inches=0.02)
    plt.close(fig)
    return out_path


def main() -> None:
    args = _parse_args()
    data_root = resolve_data_root(args.data_root)
    b3d_dir = nimble_b3d_dir(data_root)
    motion_id = _choose_motion_id(b3d_dir, args.motion_id, args.seed)

    output = args.output
    out_path = _resolve_output_path(motion_id, output, str(args.view))

    saved = render_b3d_motion_pdf(
        motion_id=motion_id,
        data_root=data_root,
        output_pdf=out_path,
        num_poses=int(args.num_poses),
        view=str(args.view),
    )
    print(f"Saved {saved} ({motion_id}, {int(args.num_poses)} poses, view={args.view})")


if __name__ == "__main__":
    main()
