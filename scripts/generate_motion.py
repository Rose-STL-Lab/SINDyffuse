from __future__ import annotations
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import argparse
import numpy as np
import torch
from common.clip_model import load_clip
from common.io import load_json
from common.paths import nimble_b3d_dir, resolve_data_root, resolve_repo_path
from common.run_logging import RunLogger, add_run_log_cli_args, run_logged_main
from common.runtime import resolve_torch_device
from diffusion.clip import clip_encode
from diffusion.config import GuidanceMode
from diffusion.model import DiffusionTransformer, GaussianDiffusionSchedule
from nimble.guidance import build_nimble_guidance
from sindy.guidance import LearnedSINDyGuidance

@torch.no_grad()
def _sample_motion(model: DiffusionTransformer, sched: GaussianDiffusionSchedule, text_ctx: torch.Tensor, text_mask: torch.Tensor, batch_size: int, seq_len: int, feat_dim: int, cfg_scale: float, device: torch.device) -> torch.Tensor:
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
        mean = recip_sqrt_alpha * (x - beta_t * eps / torch.clamp(sqrt_one_minus_ab, min=1e-08))
        if i > 0:
            var = sched.extract(sched.posterior_variance, t, x.shape)
            x = mean + torch.sqrt(torch.clamp(var, min=1e-12)) * torch.randn_like(mean)
        else:
            x = mean
    return x

def _optimize_with_guidance(motion_init: torch.Tensor, captions: List[str], mode: GuidanceMode, sindy_guidance, nimble_guidance, device: torch.device, steps: int, lr: float) -> torch.Tensor:
    motion = motion_init.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([motion], lr=float(lr))
    for _ in range(int(steps)):
        optimizer.zero_grad()
        if mode == GuidanceMode.SINDY and sindy_guidance is not None:
            loss = sindy_guidance.loss(motion, captions=captions, device=device)
        elif mode == GuidanceMode.NIMBLE and nimble_guidance is not None:
            loss = nimble_guidance.loss(motion)
        else:
            loss = torch.tensor(0.0, device=device)
        loss.backward()
        optimizer.step()
    return motion.detach()

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate motions with none|sindy|nimble guidance.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--caption', required=True)
    parser.add_argument('--out_npz', required=True)
    parser.add_argument('--guidance', choices=['none', 'sindy', 'nimble'], default='sindy')
    parser.add_argument('--sindy_checkpoint_dir', default='')
    parser.add_argument('--surrogate_checkpoint_dir', default='')
    parser.add_argument('--data_root', default='', help='HumanML3D dataset root (default: datasets/HumanML3D).')
    parser.add_argument('--train_config', default='', help='Optional train JSON; reads train.nimble_guidance when guidance=nimble.')
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--cfg_scale', type=float, default=2.5)
    parser.add_argument('--opt_steps', type=int, default=0)
    parser.add_argument('--opt_lr', type=float, default=0.001)
    parser.add_argument('--device', default='auto')
    add_run_log_cli_args(parser)
    args = parser.parse_args()

    def _run(logger: RunLogger) -> None:
        _generate(args, logger)
    run_logged_main(Path(__file__).stem, args.log_dir, _run, argv=sys.argv, no_run_log=bool(args.no_run_log))

def _generate(args: argparse.Namespace, logger: RunLogger) -> None:
    data_root = resolve_data_root(args.data_root or None)
    cache = nimble_b3d_dir(data_root)
    if not cache.is_dir():
        raise FileNotFoundError(f'Nimble B3D cache required at {cache}. Run preprocess pipeline first.')
    device = resolve_torch_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    feature_dim = int(ckpt.get('feature_dim', 0))
    if feature_dim <= 0:
        raise ValueError('Checkpoint missing feature_dim; retrain on Nimble B3D.')
    model = DiffusionTransformer(input_dim=feature_dim, max_seq_len=int(args.seq_len)).to(device)
    model.load_state_dict(ckpt['model_state'], strict=True)
    model.eval()
    sched = GaussianDiffusionSchedule(timesteps=int(args.timesteps)).to(device)
    clip_model, _ = load_clip('ViT-B/32', device=device, jit=False)
    clip_model.eval()
    text_ctx, text_mask = clip_encode(clip_model, [args.caption] * int(args.batch_size), device=device)
    motion = _sample_motion(model=model, sched=sched, text_ctx=text_ctx, text_mask=text_mask, batch_size=int(args.batch_size), seq_len=int(args.seq_len), feat_dim=feature_dim, cfg_scale=float(args.cfg_scale), device=device)
    _g = str(args.guidance).strip().lower()
    mode = GuidanceMode(_g)
    sindy_guidance = None
    nimble_guidance = None
    if mode == GuidanceMode.SINDY:
        if not args.sindy_checkpoint_dir:
            raise ValueError('--sindy_checkpoint_dir is required when guidance=sindy')
        if not args.surrogate_checkpoint_dir:
            raise ValueError('--surrogate_checkpoint_dir is required when guidance=sindy')
        sindy_guidance = LearnedSINDyGuidance(sild_dir=str(resolve_repo_path(args.sindy_checkpoint_dir)), data_root=data_root, fps=20.0, clip_model_name='ViT-B/32', surrogate_checkpoint=str(resolve_repo_path(args.surrogate_checkpoint_dir)))
    elif mode == GuidanceMode.NIMBLE:
        nimble_cfg: dict = {}
        if str(args.train_config).strip():
            train_cfg = load_json(str(args.train_config)).get('train', {})
            if isinstance(train_cfg, dict):
                block = train_cfg.get('nimble_guidance') or {}
                if isinstance(block, dict):
                    nimble_cfg = block
        nimble_guidance = build_nimble_guidance(data_root=data_root, fps=20.0, nimble_cfg=nimble_cfg, window_frames=int(args.seq_len))
    if int(args.opt_steps) > 0 and mode != GuidanceMode.NONE:
        motion = _optimize_with_guidance(motion_init=motion, captions=[args.caption] * int(args.batch_size), mode=mode, sindy_guidance=sindy_guidance, nimble_guidance=nimble_guidance, device=device, steps=int(args.opt_steps), lr=float(args.opt_lr))
    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, motion=motion.detach().cpu().numpy().astype(np.float32))
    logger.progress(f'saved: {out}')
if __name__ == '__main__':
    main()