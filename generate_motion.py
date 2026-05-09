from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

import clip  # type: ignore

from diffusion.common import resolve_torch_device
from diffusion.config import GuidanceMode, default_humanml3d_root
from diffusion.model import DiffusionTransformer, GaussianDiffusionSchedule
from osim.guidance import DeterministicOsimGuidance
from sindy.guidance import LearnedSINDyGuidance


@torch.no_grad()
def _clip_text_ctx(clip_model: torch.nn.Module, captions: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    toks = clip.tokenize(captions, truncate=True).to(device)
    x = clip_model.token_embedding(toks).type(clip_model.dtype)
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)
    x = clip_model.ln_final(x).float()
    return x, toks != 0


@torch.no_grad()
def _sample_motion(
    model: DiffusionTransformer,
    sched: GaussianDiffusionSchedule,
    text_ctx: torch.Tensor,
    text_mask: torch.Tensor,
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


def _optimize_with_guidance(
    motion_init: torch.Tensor,
    captions: List[str],
    mode: GuidanceMode,
    sindy_guidance,
    osim_guidance,
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
        elif mode == GuidanceMode.OSIM and osim_guidance is not None:
            loss = osim_guidance.loss(motion)
        else:
            loss = torch.tensor(0.0, device=device)
        loss.backward()
        optimizer.step()
    return motion.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate motions with none|sindy|osim guidance.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--guidance", choices=["none", "sindy", "osim"], default="sindy")
    parser.add_argument("--sindy_checkpoint_dir", default="")
    parser.add_argument("--data_root", default=default_humanml3d_root())
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--cfg_scale", type=float, default=2.5)
    parser.add_argument("--opt_steps", type=int, default=0)
    parser.add_argument("--opt_lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_torch_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    feature_dim = int(ckpt.get("feature_dim", 263))
    model = DiffusionTransformer(input_dim=feature_dim, max_seq_len=int(args.seq_len)).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    sched = GaussianDiffusionSchedule(timesteps=int(args.timesteps)).to(device)
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    text_ctx, text_mask = _clip_text_ctx(clip_model, [args.caption] * int(args.batch_size), device=device)
    motion = _sample_motion(
        model=model,
        sched=sched,
        text_ctx=text_ctx,
        text_mask=text_mask,
        batch_size=int(args.batch_size),
        seq_len=int(args.seq_len),
        feat_dim=feature_dim,
        cfg_scale=float(args.cfg_scale),
        device=device,
    )

    mode = GuidanceMode(str(args.guidance))
    sindy_guidance = None
    osim_guidance = None
    if mode == GuidanceMode.SINDY:
        if not args.sindy_checkpoint_dir:
            raise ValueError("--sindy_checkpoint_dir is required when guidance=sindy")
        sindy_guidance = LearnedSINDyGuidance(
            sild_dir=str(args.sindy_checkpoint_dir),
            data_root=str(args.data_root),
            fps=20.0,
            clip_model_name="ViT-B/32",
        )
    elif mode == GuidanceMode.OSIM:
        osim_guidance = DeterministicOsimGuidance(data_root=str(args.data_root), fps=20.0)

    if int(args.opt_steps) > 0 and mode != GuidanceMode.NONE:
        motion = _optimize_with_guidance(
            motion_init=motion,
            captions=[args.caption] * int(args.batch_size),
            mode=mode,
            sindy_guidance=sindy_guidance,
            osim_guidance=osim_guidance,
            device=device,
            steps=int(args.opt_steps),
            lr=float(args.opt_lr),
        )

    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, motion=motion.detach().cpu().numpy().astype(np.float32))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

