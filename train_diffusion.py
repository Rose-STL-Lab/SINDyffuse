from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import clip  # type: ignore

from common.distributed import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    log_main,
    model_state_dict,
    resolve_train_device,
    seed_all,
    setup_spawn_if_distributed,
    wrap_ddp,
)
from common.io import load_json, save_json
from common.paths import nimble_b3d_dir, resolve_data_root, resolve_repo_path
from diffusion.clip import clip_encode
from diffusion.config import GuidanceMode
from diffusion.workers import num_workers
from diffusion.registry import get_dataset
from diffusion.model import DiffusionTransformer, GaussianDiffusionSchedule
from nimble.guidance import build_nimble_guidance
from sindy.guidance import LearnedSINDyGuidance


def _collate(batch):
    motions = torch.stack([x["motion"] for x in batch], dim=0)
    captions = [x["caption"] for x in batch]
    motion_ids = [x["motion_id"] for x in batch]
    return motions, captions, motion_ids


def train(config_path: str, out_dir: str, *, preload: bool = False) -> None:
    cfg = load_json(config_path)
    dist_cfg = cfg.get("distributed") if isinstance(cfg.get("distributed"), dict) else {}
    _g_pre = str((cfg.get("train") or {}).get("guidance", "")).strip().lower()
    if _g_pre == "nimble" and int(np.__version__.split(".", maxsplit=1)[0]) >= 2:
        print(
            f"[train] ERROR: numpy {np.__version__} is incompatible with nimblephysics marker IK "
            f"(segfault). Rebuild conda env: conda env update -n sindyffuse -f environment.yaml --prune",
            flush=True,
        )
        sys.exit(1)

    use_ddp = init_distributed(distributed_cfg=dist_cfg)
    seed_all(int(cfg.get("seed", 42)))
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    dataset_name = str(data_cfg.get("dataset", "nimble"))
    data_root = resolve_data_root(data_cfg.get("data_root"))
    cache = nimble_b3d_dir(data_root)
    if not cache.is_dir():
        raise FileNotFoundError(
            f"Nimble B3D cache required at {cache}. Run preprocess_nimble.py first."
        )
    _preload = bool(preload or data_cfg.get("preload", False))
    train_ds = get_dataset(
        dataset_name,
        data_root=data_root,
        split="train",
        window_size=int(data_cfg.get("window_size", 64)),
        fps=int(data_cfg.get("fps", 20)),
        normalize=bool(data_cfg.get("normalize", True)),
        preload=_preload,
    )
    log_main(
        "[train] data.preload=True: q trajectories loaded into RAM"
        if _preload
        else "[train] data.preload=False: reading B3D windows on demand"
    )

    device = resolve_train_device(str(train_cfg.get("device", "auto")))
    per_gpu_batch = int(train_cfg.get("batch_size", 32))
    global_batch = per_gpu_batch * get_world_size()

    _nw_requested = int(train_cfg.get("num_workers", 4))
    _allow_fork = bool(train_cfg.get("allow_dataloader_fork_after_cuda", False))
    _nw = num_workers(device, _nw_requested, allow_fork_after_cuda=_allow_fork or use_ddp)
    if is_main_process() and _nw != _nw_requested:
        print(
            f"[train] num_workers={_nw_requested} disabled on CUDA (fork-after-GPU init crashes); using {_nw}. "
            f"Set train.allow_dataloader_fork_after_cuda=true with spawn for multi-worker loading.",
            flush=True,
        )

    train_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            drop_last=True,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=per_gpu_batch,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=_nw,
        drop_last=True,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )

    clip_model_name = str(model_cfg.get("clip_model_name", "ViT-B/32"))
    clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    model = DiffusionTransformer(
        input_dim=int(train_ds.feature_dim),
        model_dim=int(model_cfg.get("model_dim", 512)),
        num_layers=int(model_cfg.get("num_layers", 8)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        max_seq_len=int(data_cfg.get("window_size", 64)),
        clip_dim=int(model_cfg.get("clip_dim", 512)),
    ).to(device)
    find_unused = bool(dist_cfg.get("find_unused_parameters", False))
    model = wrap_ddp(model, find_unused_parameters=find_unused)

    sched = GaussianDiffusionSchedule(timesteps=int(model_cfg.get("timesteps", 1000))).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
    )

    _g = str(train_cfg.get("guidance", "sindy")).strip().lower()
    guidance_mode = GuidanceMode(_g)
    lambda_sindy = float(train_cfg.get("lambda_sindy", 0.1))
    lambda_nimble = float(train_cfg.get("lambda_nimble", 0.1))
    nimble_cfg = train_cfg.get("nimble_guidance") or {}
    if not isinstance(nimble_cfg, dict):
        nimble_cfg = {}
    sindy_dir_raw = str(train_cfg.get("sindy_checkpoint_dir", "")).strip()
    sindy_dir = str(resolve_repo_path(sindy_dir_raw)) if sindy_dir_raw else ""
    sindy_guidance = None
    nimble_guidance = None
    if guidance_mode == GuidanceMode.SINDY:
        if not sindy_dir:
            raise ValueError("Default guidance=sindy requires train.sindy_checkpoint_dir")
        sindy_guidance = LearnedSINDyGuidance(
            sild_dir=sindy_dir,
            data_root=data_root,
            fps=float(data_cfg.get("fps", 20.0)),
            clip_model_name=clip_model_name,
        )
    elif guidance_mode == GuidanceMode.NIMBLE:
        nimble_guidance = build_nimble_guidance(
            data_root=data_root,
            fps=float(data_cfg.get("fps", 20.0)),
            nimble_cfg=nimble_cfg if isinstance(nimble_cfg, dict) else {},
            window_frames=int(data_cfg.get("window_size", 64)),
        )

    log_main(
        f"[train] guidance={guidance_mode.value} dataset={dataset_name} "
        f"feature_dim={train_ds.feature_dim} device={device} "
        f"per_gpu_batch={per_gpu_batch} global_batch={global_batch} "
        f"world_size={get_world_size()} distributed={use_ddp} data_root={data_root}"
    )
    if guidance_mode == GuidanceMode.SINDY:
        log_main(f"[train] lambda_sindy={lambda_sindy} sindy_dir={sindy_dir!r}")
    if guidance_mode == GuidanceMode.NIMBLE:
        log_main(
            f"[train] lambda_nimble={lambda_nimble} "
            f"nimble.time_reduce={nimble_guidance.nimble_settings.time_reduce if nimble_guidance else 'n/a'} "
            f"nimble.robust={nimble_guidance.nimble_settings.robust if nimble_guidance else 'n/a'} "
            f"nimble.t_weight={nimble_guidance.nimble_settings.t_weight_schedule if nimble_guidance else 'n/a'} "
            f"nimble.max_frames={nimble_guidance.nimble_settings.max_physics_frames if nimble_guidance else 'n/a'}"
        )

    out = Path(out_dir)
    if is_main_process():
        out.mkdir(parents=True, exist_ok=True)
        cfg.setdefault("train", {})["global_batch_size"] = int(global_batch)
        cfg.setdefault("train", {})["world_size"] = int(get_world_size())
        save_json(str(out / "config_resolved.json"), cfg)

    max_steps = int(train_cfg.get("max_steps", 100000))
    log_every = int(train_cfg.get("log_every", 100))
    save_every = int(train_cfg.get("save_every", 5000))
    cond_drop_prob = float(train_cfg.get("cond_drop_prob", 0.1))

    step = 0
    epoch = 0
    while step < max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for motion, captions, _motion_ids in train_loader:
            if step >= max_steps:
                break
            step += 1
            x0 = motion.to(device, non_blocking=True)
            text_in = list(captions)
            if cond_drop_prob > 0:
                drop_mask = torch.rand((len(text_in),), device=device) < cond_drop_prob
                for i in range(len(text_in)):
                    if bool(drop_mask[i].item()):
                        text_in[i] = ""
            text_ctx, text_mask = clip_encode(clip_model, text_in, device=device)
            b = x0.shape[0]
            t = torch.randint(0, sched.timesteps, (b,), device=device).long()
            noise = torch.randn_like(x0)
            sqrt_ab = sched.extract(sched.sqrt_alphas_cumprod, t, x0.shape)
            sqrt_1mab = sched.extract(sched.sqrt_one_minus_alphas_cumprod, t, x0.shape)
            x_t = sqrt_ab * x0 + sqrt_1mab * noise
            eps_pred = model(x_t, t.float(), text_ctx=text_ctx, text_mask=text_mask)
            loss_diff = F.mse_loss(eps_pred, noise)

            loss_guidance = torch.tensor(0.0, device=device)
            guide_stats = {}
            denom = torch.clamp(sqrt_ab, min=1e-8)
            x0_pred = (x_t - sqrt_1mab * eps_pred) / denom
            if guidance_mode == GuidanceMode.SINDY and sindy_guidance is not None:
                loss_guidance = float(lambda_sindy) * sindy_guidance.loss(
                    x0_pred, captions=text_in, device=device
                )
            elif guidance_mode == GuidanceMode.NIMBLE and nimble_guidance is not None:
                raw_guide, guide_stats = nimble_guidance.loss_and_stats(x0_pred)
                t_weight = nimble_guidance.guidance_weight(t=t, total_timesteps=int(sched.timesteps))
                loss_guidance = float(lambda_nimble) * t_weight * raw_guide

            loss = loss_diff + loss_guidance
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(train_cfg.get("grad_clip", 1.0))
            )
            opt.step()

            if step % log_every == 0 and is_main_process():
                msg = (
                    f"step={step} loss={float(loss.item()):.6f} "
                    f"diff={float(loss_diff.item()):.6f} guide={float(loss_guidance.item()):.6f}"
                )
                if guidance_mode == GuidanceMode.NIMBLE and guide_stats:
                    msg += (
                        f" nimble_vel={guide_stats.get('nimble_vel', 0.0):.4f}"
                        f" nimble_acc={guide_stats.get('nimble_acc', 0.0):.4f}"
                        f" nimble_tau={guide_stats.get('nimble_torque', 0.0):.4f}"
                        f" nimble_jerk={guide_stats.get('nimble_jerk', 0.0):.4f}"
                        f" nimble_eff={guide_stats.get('nimble_effort', 0.0):.4f}"
                        f" nimble_contact_gap={guide_stats.get('nimble_contact_gap', 0.0):.4f}"
                        f" guide_scalar={guide_stats.get('nimble_guidance_scalar', 0.0):.4f}"
                    )
                print(msg, flush=True)
            if step % save_every == 0 and is_main_process():
                torch.save(
                    {
                        "model_state": model_state_dict(model),
                        "step": step,
                        "feature_dim": int(train_ds.feature_dim),
                    },
                    out / f"ckpt_step_{step:07d}.pt",
                )
        epoch += 1

    if is_main_process():
        torch.save(
            {
                "model_state": model_state_dict(model),
                "step": max_steps,
                "feature_dim": int(train_ds.feature_dim),
            },
            out / "last.pt",
        )


def main() -> None:
    setup_spawn_if_distributed()
    parser = argparse.ArgumentParser(description="Train text-conditioned diffusion model with guidance modes.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load q trajectories into RAM before training (default: read B3D on demand)",
    )
    args = parser.parse_args()
    try:
        train(config_path=str(args.config), out_dir=str(args.out_dir), preload=bool(args.preload))
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
