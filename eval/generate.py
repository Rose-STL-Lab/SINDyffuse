"""Batch motion generation for evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from common.clip_model import load_clip
from common.paths import motion_cache_dir, resolve_data_root, resolve_repo_path
from common.runtime import resolve_torch_device, set_seed
from datasets.splits import load_split_ids
from diffusion.clip import clip_encode
from diffusion.config import GuidanceMode
from diffusion.model import DiffusionTransformer, GaussianDiffusionSchedule
from eval.motion_store import GeneratedMotionRecord, save_generated_record
from sindy.guidance import LearnedSINDyGuidance


@dataclass
class GenerateConfig:
    checkpoint: str
    data_root: str = ""
    split: str = "test"
    guidance: str = "sindy"
    sindy_checkpoint_dir: str = ""
    surrogate_checkpoint_dir: str = ""
    output_dir: str = "results/eval/generated"
    variant: str = "default"
    seq_len: int = 64
    batch_size: int = 4
    timesteps: int = 1000
    cfg_scale: float = 2.5
    opt_steps: int = 0
    opt_lr: float = 1e-3
    num_samples_per_caption: int = 1
    max_motions: int = 0
    device: str = "auto"
    seed: int = 42
    motion_format: str = "mint_q"


@torch.no_grad()
def sample_motion(
    model: DiffusionTransformer,
    sched: GaussianDiffusionSchedule,
    text_ctx: torch.Tensor,
    text_mask: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    feat_dim: int,
    cfg_scale: float,
    device: torch.device,
) -> torch.Tensor:
    x = torch.randn((batch_size, seq_len, feat_dim), device=device)
    for i in reversed(range(sched.timesteps)):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        eps_c = model(x, t.float(), text_ctx=text_ctx, text_mask=text_mask)
        if float(cfg_scale) != 1.0:
            eps_u = model(x, t.float(), text_ctx=text_ctx, text_mask=text_mask, force_uncond=True)
            eps = eps_u + float(cfg_scale) * (eps_c - eps_u)
        else:
            eps = eps_c
        alpha_t = sched.extract(sched.alphas, t, x.shape)
        beta_t = sched.extract(sched.betas, t, x.shape)
        sqrt_one_minus_ab = sched.extract(sched.sqrt_one_minus_alphas_cumprod, t, x.shape)
        recip_sqrt_alpha = torch.sqrt(1.0 / alpha_t)
        mean = recip_sqrt_alpha * (x - beta_t * eps / torch.clamp(sqrt_one_minus_ab, min=1e-8))
        if i > 0:
            var = sched.extract(sched.posterior_variance, t, x.shape)
            x = mean + torch.sqrt(torch.clamp(var, min=1e-12)) * torch.randn_like(mean)
        else:
            x = mean
    return x


def optimize_with_guidance(
    motion_init: torch.Tensor,
    captions: list[str],
    *,
    mode: GuidanceMode,
    sindy_guidance,
    device: torch.device,
    steps: int,
    lr: float,
) -> torch.Tensor:
    motion = motion_init.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([motion], lr=float(lr))
    for _ in range(int(steps)):
        optimizer.zero_grad()
        if mode == GuidanceMode.SINDY and sindy_guidance is not None:
            loss = sindy_guidance.loss(motion, captions=captions, device=device)
        else:
            loss = torch.tensor(0.0, device=device)
        loss.backward()
        optimizer.step()
    return motion.detach()


def _load_captions(text_path: Path) -> list[tuple[str, str]]:
    if not text_path.is_file():
        return []
    out: list[tuple[str, str]] = []
    for line in text_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("#")
        if not parts or not parts[0].strip():
            continue
        out.append((parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""))
    return out


def iter_eval_prompts(data_root: str, split: str, *, max_motions: int = 0) -> Iterator[tuple[str, str]]:
    root = Path(resolve_data_root(data_root))
    text_dir = root / "texts" if (root / "texts").is_dir() else root
    ids = load_split_ids(root, split)
    if max_motions > 0:
        ids = ids[: max_motions]
    for motion_id in ids:
        captions = _load_captions(text_dir / f"{motion_id}.txt")
        if not captions:
            continue
        caption, _ = random.choice(captions)
        yield motion_id, caption


def generate_batch(cfg: GenerateConfig) -> list[GeneratedMotionRecord]:
    set_seed(cfg.seed)
    data_root = resolve_data_root(cfg.data_root or None)
    cache = motion_cache_dir(data_root)
    if not cache.is_dir():
        raise FileNotFoundError(f"Motion cache required at {cache}. Run preprocess_mint.py first.")

    device = resolve_torch_device(cfg.device)
    ckpt = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    feature_dim = int(ckpt.get("feature_dim", 0))
    if feature_dim <= 0:
        raise ValueError("Checkpoint missing feature_dim; retrain on MinT cache.")

    model = DiffusionTransformer(input_dim=feature_dim, max_seq_len=int(cfg.seq_len)).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    sched = GaussianDiffusionSchedule(timesteps=int(cfg.timesteps)).to(device)
    clip_model, _ = load_clip("ViT-B/32", device=device, jit=False)
    clip_model.eval()

    mode = GuidanceMode(str(cfg.guidance).strip().lower())
    sindy_guidance = None
    if mode == GuidanceMode.SINDY:
        if not cfg.sindy_checkpoint_dir or not cfg.surrogate_checkpoint_dir:
            raise ValueError("sindy_checkpoint_dir and surrogate_checkpoint_dir required for guidance=sindy")
        sindy_guidance = LearnedSINDyGuidance(
            sild_dir=str(resolve_repo_path(cfg.sindy_checkpoint_dir)),
            data_root=data_root,
            fps=20.0,
            clip_model_name="ViT-B/32",
            surrogate_checkpoint=str(resolve_repo_path(cfg.surrogate_checkpoint_dir)),
        )

    out_dir = resolve_repo_path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[GeneratedMotionRecord] = []

    prompts = list(iter_eval_prompts(data_root, cfg.split, max_motions=cfg.max_motions))
    for sample_idx, (motion_id, caption) in enumerate(prompts):
        for rep in range(cfg.num_samples_per_caption):
            text_ctx, text_mask = clip_encode(clip_model, [caption], device=device)
            motion = sample_motion(
                model=model,
                sched=sched,
                text_ctx=text_ctx,
                text_mask=text_mask,
                batch_size=1,
                seq_len=int(cfg.seq_len),
                feat_dim=feature_dim,
                cfg_scale=float(cfg.cfg_scale),
                device=device,
            )
            if int(cfg.opt_steps) > 0 and mode != GuidanceMode.NONE:
                motion = optimize_with_guidance(
                    motion,
                    [caption],
                    mode=mode,
                    sindy_guidance=sindy_guidance,
                    device=device,
                    steps=int(cfg.opt_steps),
                    lr=float(cfg.opt_lr),
                )

            motion_np = motion.detach().cpu().numpy()[0].astype(np.float32)
            record = GeneratedMotionRecord(
                sample_id=f"{motion_id}_s{sample_idx}_r{rep}",
                motion_id=motion_id,
                caption=caption,
                motion=motion_np,
                length=int(motion_np.shape[0]),
                format=cfg.motion_format,
                variant=cfg.variant,
                seed=cfg.seed + sample_idx * 100 + rep,
                metadata={
                    "guidance": cfg.guidance,
                    "cfg_scale": cfg.cfg_scale,
                    "opt_steps": cfg.opt_steps,
                },
            )
            save_generated_record(out_dir / record.sample_id, record)
            records.append(record)
    return records
