#!/usr/bin/env python3
"""Render a stacked PDF: SINDyffuse (Ours) Rajagopal montage (top) vs MDM (bottom)."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from textwrap import wrap
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.paths import humanml3d_text_dir, nimble_b3d_dir, resolve_data_root, resolve_repo_path, results_dir
from nimble.ik import fit_q
from nimble.skeleton_registry import load_skeleton
from nimble.viz import (
    compute_height_offset,
    draw_rajagopal_montage,
    figure_content_bbox,
    fit_figure_to_projected_content,
    rajagopal_mesh_specs,
)
from nimble.viz import _project_bounds_corners_2d  # noqa: PLC2701

_WENHAO_HANDOFF = Path(
    "/mnt/Wenhao/artifacts/cross_pvc_table_handoff_20260727/source_artifacts/biomech_ai"
)
DEFAULT_MDM_CORPUS_ROOT = _WENHAO_HANDOFF / "generated_corpora"
DEFAULT_MDM_RECORDS = DEFAULT_MDM_CORPUS_ROOT / "shared_records" / "selected_records.json"
DEFAULT_SINDYFFUSE_JOINTS = DEFAULT_MDM_CORPUS_ROOT / "SINDyffuse" / "joints.npy"
DEFAULT_MDM_JOINTS = DEFAULT_MDM_CORPUS_ROOT / "MDM" / "joints.npy"
DEFAULT_MDM_MODEL_PATH = _WENHAO_HANDOFF / "models" / "baselines" / "MDM" / "model000750000.pt"
DEFAULT_MDM_ARGS_JSON = (
    _WENHAO_HANDOFF
    / "models"
    / "config_sources"
    / "baselines"
    / "motion-diffusion-model"
    / "save"
    / "humanml_enc_512_50steps"
    / "args.json"
)

MOTION_ID: str | None = None
DATA_ROOT: str | None = None
OUTPUT_PDF: str | None = None
NUM_POSES: int = 6
RANDOM_SEED: int | None = None
CAPTION_FONT_SIZE: float = 8.0
ROW_LABEL_FONT_SIZE: float = 9.0
ROW_LABEL_GAP_PX: float = 2.0
CAPTION_ABOVE_LABEL_PX: float = 10.0
ROW_GAP_INCHES: float = 0.08  # inches between top mesh bottom and MDM label / bottom mesh
SAVE_PAD_INCHES: float = 0.015
FIG_DPI: int = 300
VIEW: str = "default"
MDM_SEED: int = 42
MDM_GUIDANCE: float = 2.5
MDM_MOTION_LENGTH: float = 3.2

OURS_ROW_LABEL = "SINDyffuse (Ours)"
MDM_ROW_LABEL = "MDM"

_IK_CONFIG = {
    "scale_bodies": False,
    "line_search": True,
    "log_output": False,
    "bidirectional": True,
    "scale_first_frame": True,
    "repair_poor_fk_frames": True,
    "fk_loss_repair_threshold": 0.01,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a stacked Rajagopal mesh PDF: SINDyffuse (Ours) on top, "
            "MDM retargeted via IK on the bottom."
        ),
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
        help="Output PDF path (default: results/viz/{motion_id}_vs_mdm.pdf)",
    )
    parser.add_argument(
        "--num_poses",
        type=int,
        default=NUM_POSES,
        help="Number of evenly spaced poses per row",
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
        help='Camera preset for both rows: "default" oblique top-down, "flat" side view',
    )
    parser.add_argument(
        "--mdm_corpus_root",
        type=Path,
        default=DEFAULT_MDM_CORPUS_ROOT,
        help="Root containing MDM/joints.npy (precomputed corpus)",
    )
    parser.add_argument(
        "--mdm_records",
        type=Path,
        default=DEFAULT_MDM_RECORDS,
        help="selected_records.json mapping source_id to corpus index",
    )
    parser.add_argument(
        "--sindyffuse_joints",
        type=Path,
        default=DEFAULT_SINDYFFUSE_JOINTS,
        help="Precomputed SINDyffuse joints.npy [N,T,22,3]",
    )
    parser.add_argument(
        "--mdm_joints",
        type=Path,
        default=DEFAULT_MDM_JOINTS,
        help="Precomputed MDM joints.npy [N,T,22,3]",
    )
    parser.add_argument(
        "--generate_mdm",
        action="store_true",
        help="Sample MDM joints live from the motion text caption (requires MDM install)",
    )
    parser.add_argument(
        "--mdm_root",
        type=Path,
        default=None,
        help="Path to motion-diffusion-model repository",
    )
    parser.add_argument(
        "--mdm_runtime",
        type=Path,
        default=None,
        help="Runtime directory with dataset/ and glove/ symlinks (MDM cwd)",
    )
    parser.add_argument(
        "--mdm_model_path",
        type=Path,
        default=DEFAULT_MDM_MODEL_PATH,
        help="MDM checkpoint .pt file",
    )
    parser.add_argument(
        "--mdm_args_json",
        type=Path,
        default=DEFAULT_MDM_ARGS_JSON,
        help="Training args.json for the MDM checkpoint (if not beside model_path)",
    )
    parser.add_argument(
        "--mdm_seed",
        type=int,
        default=MDM_SEED,
        help="Random seed for live MDM sampling",
    )
    parser.add_argument(
        "--mdm_guidance",
        type=float,
        default=MDM_GUIDANCE,
        help="Classifier-free guidance scale for live MDM sampling",
    )
    parser.add_argument(
        "--mdm_motion_length",
        type=float,
        default=MDM_MOTION_LENGTH,
        help="Motion length in seconds for live MDM sampling (3.2 s -> 64 frames @ 20 fps)",
    )
    return parser.parse_args()


def _list_corpus_motion_ids(records_path: Path) -> list[str]:
    lookup = _load_corpus_index_lookup(str(records_path.resolve()))
    return sorted(lookup)


def _choose_motion_id(
    b3d_dir: Path,
    motion_id: str | None,
    seed: int | None,
    *,
    records_path: Path,
) -> str:
    corpus_ids = _list_corpus_motion_ids(records_path)
    if not corpus_ids:
        raise FileNotFoundError(f"No motion ids found in corpus records: {records_path}")

    if motion_id is not None and str(motion_id).strip():
        chosen = str(motion_id).strip()
        if _lookup_corpus_index(chosen, records_path) is None:
            raise KeyError(
                f"Motion id {chosen!r} is not in the precomputed corpus at {records_path}. "
                "Pick an id from selected_records.json or pass --generate_mdm."
            )
        return chosen

    rng = random.Random(seed)
    return rng.choice(corpus_ids)


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


def _default_output_path(motion_id: str) -> Path:
    return results_dir() / "viz" / f"{motion_id}_vs_mdm.pdf"


def _resolve_output_path(motion_id: str, output: str | None) -> Path:
    if output is None or str(output).strip() == "":
        return _default_output_path(motion_id)
    return resolve_repo_path(str(output))


@lru_cache(maxsize=1)
def _load_corpus_index_lookup(records_path: str) -> dict[str, int]:
    path = Path(records_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing MDM records file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError(f"Unsupported selected_records mapping keys: {sorted(payload)[:20]}")
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"Unsupported selected_records type: {type(payload).__name__}")

    lookup: dict[str, int] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        for key in ("source_id", "key", "motion_id"):
            value = record.get(key)
            if value is None:
                continue
            token = str(value).strip()
            if token:
                lookup.setdefault(token, int(index))
    return lookup


def _lookup_corpus_index(motion_id: str, records_path: Path) -> int | None:
    lookup = _load_corpus_index_lookup(str(records_path.resolve()))
    return lookup.get(str(motion_id).strip())


def load_precomputed_joints(
    motion_id: str,
    *,
    method: str,
    joints_path: Path,
    records_path: Path,
) -> np.ndarray:
    """Load generated joints [T,22,3] for a motion_id from the Wenhao precomputed corpus."""
    index = _lookup_corpus_index(motion_id, records_path)
    if index is None:
        raise KeyError(
            f"Motion id {motion_id!r} is not in the {method} corpus records at {records_path}. "
            "Try another --motion_id or pass --generate_mdm for live MDM sampling."
        )
    path = joints_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing precomputed {method} joints: {path}")
    corpus = np.load(path, mmap_mode="r")
    if corpus.ndim != 4 or corpus.shape[2:] != (22, 3):
        raise ValueError(f"Expected {method} joints [N,T,22,3], got {corpus.shape}")
    if index >= corpus.shape[0]:
        raise IndexError(f"{method}: corpus index {index} >= joint count {corpus.shape[0]}")
    joints = np.asarray(corpus[index], dtype=np.float64)
    if not np.isfinite(joints).all():
        raise ValueError(f"{method} joints for {motion_id} (index {index}) contain non-finite values")
    return joints


def retarget_joints_to_rajagopal_q(joints: np.ndarray) -> np.ndarray:
    """Fit HumanML3D joints [T,22,3] to Rajagopal q [T,37] via IK."""
    poses = np.asarray(joints, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (22, 3):
        raise ValueError(f"Expected joints [T,22,3], got {poses.shape}")

    parsed, spec = load_skeleton("rajagopal")
    skeleton = parsed.skeleton
    q_dof_t, _stats = fit_q(
        poses,
        skeleton,
        ik_mapping=spec.ik_mapping,
        **_IK_CONFIG,
    )
    q = np.asarray(q_dof_t, dtype=np.float64).T
    if q.shape[1] != 37:
        raise ValueError(f"Expected q [T,37], got {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError("Retargeted q contains non-finite values")
    return q


def _prepare_mdm_model_dir(model_path: Path, args_json: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing MDM checkpoint: {model_path}")

    local_args = model_path.parent / "args.json"
    if local_args.is_file():
        return model_path, None

    args_json = args_json.expanduser().resolve()
    if not args_json.is_file():
        raise FileNotFoundError(
            f"Missing args.json beside {model_path} and no --mdm_args_json found at {args_json}"
        )

    tmp = tempfile.TemporaryDirectory(prefix="mdm_model_")
    staging = Path(tmp.name)
    staged_model = staging / model_path.name
    os.symlink(model_path, staged_model)
    shutil.copy2(args_json, staging / "args.json")
    return staged_model, tmp


def generate_mdm_joints_live(
    *,
    text_prompt: str,
    mdm_root: Path,
    mdm_runtime: Path,
    model_path: Path,
    args_json: Path,
    seed: int,
    guidance: float,
    motion_length: float,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Sample one MDM motion from a text prompt and return joints [T,22,3]."""
    mdm_root = mdm_root.expanduser().resolve()
    mdm_runtime = mdm_runtime.expanduser().resolve()
    if not mdm_root.is_dir():
        raise FileNotFoundError(
            f"MDM repository not found at {mdm_root}. "
            "Clone motion-diffusion-model and pass --mdm_root, or omit --generate_mdm "
            "to use the precomputed Wenhao corpus."
        )
    if not mdm_runtime.is_dir():
        raise FileNotFoundError(
            f"MDM runtime directory not found at {mdm_runtime}. "
            "Create it with symlinks to HumanML3D dataset/ and glove/, then pass --mdm_runtime."
        )

    staged_model, staging_tmp = _prepare_mdm_model_dir(model_path, args_json)
    out_dir = cache_dir
    cleanup_out = False
    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="mdm_sample_", dir=str(results_dir() / "viz" / "_mdm_cache")))
        cleanup_out = True
    else:
        out_dir = out_dir.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    old_cwd = os.getcwd()
    old_argv = sys.argv[:]
    old_path = sys.path[:]
    try:
        if str(mdm_root) not in sys.path:
            sys.path.insert(0, str(mdm_root))
        os.chdir(mdm_runtime)

        sys.argv = [
            "visualize_b3d_vs_mdm_motion.py",
            "--model_path",
            str(staged_model),
            "--output_dir",
            str(out_dir),
            "--text_prompt",
            text_prompt,
            "--num_samples",
            "1",
            "--num_repetitions",
            "1",
            "--seed",
            str(seed),
            "--guidance_param",
            str(guidance),
            "--motion_length",
            str(motion_length),
        ]

        import torch
        from data_loaders.get_data import get_dataset_loader
        from data_loaders.humanml.scripts.motion_process import recover_from_ric
        from data_loaders.tensors import collate
        from utils import dist_util
        from utils.fixseed import fixseed
        from utils.model_util import create_model_and_diffusion, load_saved_model
        from utils.parser_util import generate_args
        from utils.sampler_util import AutoRegressiveSampler, ClassifierFreeSampleModel

        args = generate_args()
        fixseed(args.seed)
        max_frames = 196 if args.dataset in ["kit", "humanml"] else 60
        fps = 12.5 if args.dataset == "kit" else 20
        n_frames = min(max_frames, int(args.motion_length * fps))
        args.batch_size = 1
        args.num_samples = 1
        args.num_repetitions = 1

        data = get_dataset_loader(
            name=args.dataset,
            batch_size=args.batch_size,
            num_frames=max_frames,
            split="test",
            hml_mode="text_only",
            fixed_len=n_frames,
            pred_len=args.pred_len if args.pred_len > 0 else n_frames,
            device="cpu",
            autoregressive=args.autoregressive,
        )

        model, diffusion = create_model_and_diffusion(args, data)
        sample_fn = diffusion.p_sample_loop
        if args.autoregressive:
            sample_fn = AutoRegressiveSampler(args, sample_fn, n_frames).sample

        load_saved_model(model, args.model_path, use_avg=args.use_ema)
        if args.guidance_param != 1:
            model = ClassifierFreeSampleModel(model)
        model.to(dist_util.dev())
        model.eval()

        motion_shape = (1, model.njoints, model.nfeats, n_frames)
        collate_args = [{"inp": torch.zeros(n_frames), "tokens": None, "lengths": n_frames, "text": text_prompt}]
        _, model_kwargs = collate(collate_args)
        model_kwargs["y"] = {
            key: val.to(dist_util.dev()) if torch.is_tensor(val) else val
            for key, val in model_kwargs["y"].items()
        }
        if args.guidance_param != 1:
            model_kwargs["y"]["scale"] = torch.ones(1, device=dist_util.dev()) * args.guidance_param
        model_kwargs["y"]["text_embed"] = model.encode_text(model_kwargs["y"]["text"])

        with torch.no_grad():
            sample = sample_fn(
                model,
                motion_shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

        if model.data_rep != "hml_vec":
            raise ValueError(f"Expected MDM hml_vec data_rep, got {model.data_rep}")

        hml_modelnorm = sample.detach().cpu().permute(0, 2, 3, 1).float()
        hml_raw = data.dataset.t2m_dataset.inv_transform(hml_modelnorm).float()
        n_joints = 22 if sample.shape[1] == 263 else 21
        joints = recover_from_ric(hml_raw, n_joints)
        joints_np = joints.view(-1, *joints.shape[2:]).cpu().numpy().astype(np.float64)[0]
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
        sys.path = old_path
        if staging_tmp is not None:
            staging_tmp.cleanup()
        if cleanup_out and out_dir is not None and out_dir.is_dir():
            shutil.rmtree(out_dir, ignore_errors=True)

    if joints_np.ndim != 3 or joints_np.shape[1:] != (22, 3):
        raise ValueError(f"Live MDM sampling returned unexpected shape {joints_np.shape}")
    return joints_np


def _render_q_row(
    ax: Any,
    skeleton: Any,
    mesh_specs: list[Any],
    q: np.ndarray,
    *,
    num_poses: int,
    view: str,
) -> tuple[np.ndarray, np.ndarray]:
    frame_ids = _frame_indices(int(q.shape[0]), num_poses)
    selected_q = [q[frame_idx] for frame_idx in frame_ids]
    height_offset = compute_height_offset(skeleton, selected_q, mesh_specs)
    mesh_mins, mesh_maxs = draw_rajagopal_montage(
        ax,
        skeleton,
        selected_q,
        mesh_specs,
        height_offset=height_offset,
        view=view,
    )
    ax.set_axis_off()
    return mesh_mins, mesh_maxs


def _projected_mesh_width_px(
    fig: Any,
    ax: Any,
    mins: np.ndarray,
    maxs: np.ndarray,
) -> float:
    fig.canvas.draw()
    pts = _project_bounds_corners_2d(ax, mins, maxs)
    return float(pts[:, 0].max() - pts[:, 0].min())


def _setup_row_figure(
    skeleton: Any,
    mesh_specs: list[Any],
    q: np.ndarray,
    *,
    fig_width: float,
    num_poses: int,
    view: str,
) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    fig = plt.figure(figsize=(fig_width, 3.0), dpi=FIG_DPI)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    mins, maxs = _render_q_row(
        ax,
        skeleton,
        mesh_specs,
        q,
        num_poses=num_poses,
        view=view,
    )
    label_margin = ROW_LABEL_FONT_SIZE / 72.0 * 1.15
    fit_figure_to_projected_content(
        fig,
        ax,
        mins,
        maxs,
        top_margin=0.03 + label_margin,
        bottom_margin=0.03,
    )
    return fig, ax, mins, maxs


def _measure_row_mesh_width_px(
    skeleton: Any,
    mesh_specs: list[Any],
    q: np.ndarray,
    *,
    fig_width: float,
    num_poses: int,
    view: str,
) -> float:
    fig, ax, mins, maxs = _setup_row_figure(
        skeleton,
        mesh_specs,
        q,
        fig_width=fig_width,
        num_poses=num_poses,
        view=view,
    )
    width_px = _projected_mesh_width_px(fig, ax, mins, maxs)
    plt.close(fig)
    return width_px


def _matched_row_fig_widths(
    skeleton: Any,
    mesh_specs: list[Any],
    q_rows: Sequence[np.ndarray],
    *,
    base_fig_width: float,
    num_poses: int,
    view: str,
) -> list[float]:
    """Pick per-row fig widths so each row's mesh projects to the same pixel width."""
    widths_px = [
        _measure_row_mesh_width_px(
            skeleton,
            mesh_specs,
            q,
            fig_width=base_fig_width,
            num_poses=num_poses,
            view=view,
        )
        for q in q_rows
    ]
    target_px = max(widths_px)
    return [base_fig_width * (target_px / max(width_px, 1e-6)) for width_px in widths_px]


def _render_row_rgba(
    skeleton: Any,
    mesh_specs: list[Any],
    q: np.ndarray,
    *,
    fig_width: float,
    num_poses: int,
    view: str,
    row_label: str,
) -> tuple[np.ndarray, int]:
    """Render one montage row; return RGBA image and label band height in pixels."""
    from PIL import Image

    fig, ax, mins, maxs = _setup_row_figure(
        skeleton,
        mesh_specs,
        q,
        fig_width=fig_width,
        num_poses=num_poses,
        view=view,
    )
    label = _add_row_label(fig, ax, row_label, mins, maxs)
    crop = figure_content_bbox(fig, ax, mins, maxs, extra_artists=[label], pad_pixels=2.0)
    label_band_px = int(round(ROW_LABEL_FONT_SIZE * FIG_DPI / 72.0 * 1.25 + ROW_LABEL_GAP_PX + 4.0))

    buf = BytesIO()
    fig.savefig(
        buf,
        format="png",
        bbox_inches=crop,
        pad_inches=0.01,
        dpi=FIG_DPI,
        transparent=True,
    )
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGBA")), label_band_px


def _render_caption_rgba(caption: str, *, fig_width: float) -> np.ndarray:
    from PIL import Image

    wrap_cols = max(28, int(fig_width * 0.96 * 7))
    wrapped = "\n".join(wrap(str(caption).strip(), width=wrap_cols))
    lines = max(1, wrapped.count("\n") + 1)
    fig_h = max(0.35, 0.18 + lines * (CAPTION_FONT_SIZE / 72.0) * 1.15)
    fig = plt.figure(figsize=(fig_width, fig_h), dpi=FIG_DPI)
    fig.text(
        0.5,
        0.5,
        wrapped,
        ha="center",
        va="center",
        fontsize=CAPTION_FONT_SIZE,
        fontweight="bold",
        transform=fig.transFigure,
    )
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02, dpi=FIG_DPI, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGBA"))


def _paste_centered(canvas: Any, image: np.ndarray, y: int) -> None:
    from PIL import Image

    layer = Image.fromarray(image)
    x = max(0, (canvas.size[0] - layer.size[0]) // 2)
    canvas.paste(layer, (x, y), layer)


def _scale_row_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    """Downscale a row image to target width; never upscale (avoids blur)."""
    from PIL import Image

    current_width = int(image.shape[1])
    if current_width <= target_width:
        return image
    scale = target_width / float(current_width)
    target_height = max(1, int(round(image.shape[0] * scale)))
    resized = Image.fromarray(image).resize((target_width, target_height), Image.Resampling.LANCZOS)
    return np.asarray(resized)


def _scale_row_mesh_preserve_label(
    row_rgba: np.ndarray,
    label_band_px: int,
    target_width: int,
) -> np.ndarray:
    """Scale only the mesh portion; keep the label band at native resolution/font size."""
    from PIL import Image

    label_band_px = max(1, min(int(label_band_px), int(row_rgba.shape[0])))
    label_band = row_rgba[:label_band_px]
    mesh_band = row_rgba[label_band_px:]
    mesh_scaled = _scale_row_to_width(mesh_band, target_width)
    label_img = Image.fromarray(label_band)
    mesh_img = Image.fromarray(mesh_scaled)

    canvas = Image.new("RGBA", (target_width, label_band_px + mesh_scaled.shape[0]), (255, 255, 255, 0))
    label_x = max(0, (target_width - label_img.size[0]) // 2)
    canvas.paste(label_img, (label_x, 0), label_img)
    mesh_x = max(0, (target_width - mesh_img.size[0]) // 2)
    canvas.paste(mesh_img, (mesh_x, label_band_px), mesh_img)
    return np.asarray(canvas)


def _compose_stacked_pdf(
    *,
    top_rgba: np.ndarray,
    top_label_band_px: int,
    bot_rgba: np.ndarray,
    bot_label_band_px: int,
    caption: str | None,
    output_pdf: Path,
) -> None:
    from PIL import Image

    gap_px = max(1, int(ROW_GAP_INCHES * FIG_DPI))
    caption_gap_px = max(1, int(CAPTION_ABOVE_LABEL_PX))

    target_width = max(int(top_rgba.shape[1]), int(bot_rgba.shape[1]))
    top_rgba = _scale_row_mesh_preserve_label(top_rgba, top_label_band_px, target_width)
    bot_rgba = _scale_row_mesh_preserve_label(bot_rgba, bot_label_band_px, target_width)

    width = target_width
    caption_rgba = _render_caption_rgba(caption, fig_width=width / FIG_DPI) if caption else None

    caption_block = 0
    if caption_rgba is not None:
        caption_block = int(caption_rgba.shape[0]) + caption_gap_px

    total_h = caption_block + int(top_rgba.shape[0]) + gap_px + int(bot_rgba.shape[0])
    canvas = Image.new("RGBA", (width, total_h), (255, 255, 255, 255))

    y = 0
    if caption_rgba is not None:
        _paste_centered(canvas, caption_rgba, y)
        y += int(caption_rgba.shape[0]) + caption_gap_px

    _paste_centered(canvas, top_rgba, y)
    y += int(top_rgba.shape[0]) + gap_px
    _paste_centered(canvas, bot_rgba, y)

    canvas.convert("RGB").save(output_pdf, "PDF", resolution=FIG_DPI)


def _display_y_to_figure(fig: Any, y_display: float) -> float:
    return float(fig.transFigure.inverted().transform((0.0, y_display))[1])


def _mesh_top_y_figure(
    fig: Any,
    ax: Any,
    mins: np.ndarray,
    maxs: np.ndarray,
    *,
    gap_px: float = ROW_LABEL_GAP_PX,
) -> float:
    fig.canvas.draw()
    pts = _project_bounds_corners_2d(ax, mins, maxs)
    top_display = float(pts[:, 1].max()) + float(gap_px)
    return _display_y_to_figure(fig, top_display)


def _add_row_label(
    fig: Any,
    ax: Any,
    label: str,
    mins: np.ndarray,
    maxs: np.ndarray,
) -> Any:
    y = _mesh_top_y_figure(fig, ax, mins, maxs)
    text = fig.text(
        0.5,
        y,
        label,
        ha="center",
        va="bottom",
        fontsize=ROW_LABEL_FONT_SIZE,
        transform=fig.transFigure,
    )
    fig.canvas.draw()
    return text


def render_b3d_vs_mdm_pdf(
    *,
    motion_id: str,
    data_root: str | Path,
    output_pdf: str | Path,
    num_poses: int = NUM_POSES,
    view: str = VIEW,
    mdm_joints: np.ndarray | None = None,
    ours_joints: np.ndarray | None = None,
    generate_mdm: bool = False,
    mdm_records: Path | None = None,
    sindyffuse_joints_path: Path | None = None,
    mdm_joints_path: Path | None = None,
    mdm_root: Path | None = None,
    mdm_runtime: Path | None = None,
    mdm_model_path: Path | None = None,
    mdm_args_json: Path | None = None,
    mdm_seed: int = MDM_SEED,
    mdm_guidance: float = MDM_GUIDANCE,
    mdm_motion_length: float = MDM_MOTION_LENGTH,
) -> Path:
    data_root = Path(resolve_data_root(str(data_root)))
    records_path = mdm_records or DEFAULT_MDM_RECORDS
    caption = _load_caption(data_root, motion_id)

    if ours_joints is None:
        ours_joints = load_precomputed_joints(
            motion_id,
            method="SINDyffuse",
            joints_path=sindyffuse_joints_path or DEFAULT_SINDYFFUSE_JOINTS,
            records_path=records_path,
        )

    if mdm_joints is None:
        if generate_mdm:
            prompt = caption
            if not prompt:
                raise ValueError(
                    f"Cannot generate MDM motion for {motion_id}: missing text caption under {data_root}"
                )
            if mdm_root is None or mdm_runtime is None:
                raise ValueError("--generate_mdm requires both --mdm_root and --mdm_runtime")
            mdm_joints = generate_mdm_joints_live(
                text_prompt=prompt,
                mdm_root=mdm_root,
                mdm_runtime=mdm_runtime,
                model_path=mdm_model_path or DEFAULT_MDM_MODEL_PATH,
                args_json=mdm_args_json or DEFAULT_MDM_ARGS_JSON,
                seed=int(mdm_seed),
                guidance=float(mdm_guidance),
                motion_length=float(mdm_motion_length),
            )
        else:
            mdm_joints = load_precomputed_joints(
                motion_id,
                method="MDM",
                joints_path=mdm_joints_path or DEFAULT_MDM_JOINTS,
                records_path=records_path,
            )

    q_ours = retarget_joints_to_rajagopal_q(ours_joints)
    q_mdm = retarget_joints_to_rajagopal_q(mdm_joints)

    parsed, _spec = load_skeleton("rajagopal")
    skeleton = parsed.skeleton
    mesh_specs = rajagopal_mesh_specs(skeleton)
    if not mesh_specs:
        raise RuntimeError("Rajagopal skeleton has no mesh shape nodes to render")

    n_poses = max(1, int(num_poses))
    base_fig_width = max(8.0, 1.6 + 1.4 * n_poses)
    fig_width_top, fig_width_bot = _matched_row_fig_widths(
        skeleton,
        mesh_specs,
        [q_ours, q_mdm],
        base_fig_width=base_fig_width,
        num_poses=n_poses,
        view=view,
    )

    top_rgba, top_label_band_px = _render_row_rgba(
        skeleton,
        mesh_specs,
        q_ours,
        fig_width=fig_width_top,
        num_poses=n_poses,
        view=view,
        row_label=OURS_ROW_LABEL,
    )
    bot_rgba, bot_label_band_px = _render_row_rgba(
        skeleton,
        mesh_specs,
        q_mdm,
        fig_width=fig_width_bot,
        num_poses=n_poses,
        view=view,
        row_label=MDM_ROW_LABEL,
    )

    out_path = Path(output_pdf).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _compose_stacked_pdf(
        top_rgba=top_rgba,
        top_label_band_px=top_label_band_px,
        bot_rgba=bot_rgba,
        bot_label_band_px=bot_label_band_px,
        caption=caption,
        output_pdf=out_path,
    )
    return out_path


def main() -> None:
    args = _parse_args()
    data_root = resolve_data_root(args.data_root)
    motion_id = _choose_motion_id(
        nimble_b3d_dir(data_root),
        args.motion_id,
        args.seed,
        records_path=args.mdm_records,
    )
    out_path = _resolve_output_path(motion_id, args.output)

    saved = render_b3d_vs_mdm_pdf(
        motion_id=motion_id,
        data_root=data_root,
        output_pdf=out_path,
        num_poses=int(args.num_poses),
        view=str(args.view),
        generate_mdm=bool(args.generate_mdm),
        mdm_records=args.mdm_records,
        sindyffuse_joints_path=args.sindyffuse_joints,
        mdm_joints_path=args.mdm_joints,
        mdm_root=args.mdm_root,
        mdm_runtime=args.mdm_runtime,
        mdm_model_path=args.mdm_model_path,
        mdm_args_json=args.mdm_args_json,
        mdm_seed=int(args.mdm_seed),
        mdm_guidance=float(args.mdm_guidance),
        mdm_motion_length=float(args.mdm_motion_length),
    )
    source = "live MDM" if args.generate_mdm else "precomputed corpus"
    print(
        f"Saved {saved} ({motion_id}, {int(args.num_poses)} poses/row, "
        f"view={args.view}, mdm={source})"
    )


if __name__ == "__main__":
    main()
