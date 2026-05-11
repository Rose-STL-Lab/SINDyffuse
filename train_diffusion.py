from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import clip  # type: ignore

from diffusion.common import load_json, resolve_data_root, resolve_torch_device, save_json, set_seed
from diffusion.config import GuidanceMode
from diffusion.data_registry import get_text_motion_dataset
from diffusion.model import DiffusionTransformer, GaussianDiffusionSchedule
from osim.guidance import DeterministicOsimGuidance, OsimGuidanceWeights, OsimOracleConfig
from sindy.guidance import LearnedSINDyGuidance


def _collate(batch):
    motions = torch.stack([x["motion"] for x in batch], dim=0)
    captions = [x["caption"] for x in batch]
    sample_ids = [x["sample_id"] for x in batch]
    return motions, captions, sample_ids


@torch.no_grad()
def _clip_text_ctx(clip_model: torch.nn.Module, captions: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    toks = clip.tokenize(captions, truncate=True).to(device)
    x = clip_model.token_embedding(toks).type(clip_model.dtype)
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)
    x = clip_model.ln_final(x).float()
    mask = toks != 0
    return x, mask


def train(config_path: str, out_dir: str) -> None:
    cfg = load_json(config_path)
    set_seed(int(cfg.get("seed", 42)))
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    dataset_name = str(data_cfg.get("dataset", "humanml3d"))
    data_root = resolve_data_root(data_cfg.get("data_root"))
    train_ds = get_text_motion_dataset(dataset_name, data_root=data_root, split="train", window_size=int(data_cfg.get("window_size", 64)), fps=int(data_cfg.get("fps", 20)), normalize=bool(data_cfg.get("normalize", True)))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        drop_last=True,
        collate_fn=_collate,
    )

    device = resolve_torch_device(str(train_cfg.get("device", "auto")))
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
    sched = GaussianDiffusionSchedule(timesteps=int(model_cfg.get("timesteps", 1000))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 2e-4)), weight_decay=float(train_cfg.get("weight_decay", 1e-2)))

    guidance_mode = GuidanceMode(str(train_cfg.get("guidance", "sindy")))
    lambda_sindy = float(train_cfg.get("lambda_sindy", 0.1))
    lambda_osim = float(train_cfg.get("lambda_osim", 0.1))
    osim_cfg = train_cfg.get("osim_oracle", {}) if isinstance(train_cfg.get("osim_oracle", {}), dict) else {}
    sindy_dir = str(train_cfg.get("sindy_checkpoint_dir", "")).strip()
    sindy_guidance = None
    osim_guidance = None
    if guidance_mode == GuidanceMode.SINDY:
        if not sindy_dir:
            raise ValueError("Default guidance=sindy requires train.sindy_checkpoint_dir")
        sindy_guidance = LearnedSINDyGuidance(
            sild_dir=sindy_dir,
            data_root=data_root,
            fps=float(data_cfg.get("fps", 20.0)),
            clip_model_name=clip_model_name,
        )
    elif guidance_mode == GuidanceMode.OSIM:
        wcfg = osim_cfg.get("weights", {}) if isinstance(osim_cfg.get("weights", {}), dict) else {}
        ocfg = osim_cfg.get("oracle", {}) if isinstance(osim_cfg.get("oracle", {}), dict) else {}
        osim_guidance = DeterministicOsimGuidance(
            data_root=data_root,
            fps=float(data_cfg.get("fps", 20.0)),
            weights=OsimGuidanceWeights(
                lambda_vel=float(wcfg.get("lambda_vel", 0.1)),
                lambda_acc=float(wcfg.get("lambda_acc", 0.1)),
                lambda_torque=float(wcfg.get("lambda_torque", 0.1)),
                lambda_jerk=float(wcfg.get("lambda_jerk", 0.1)),
                lambda_effort=float(wcfg.get("lambda_effort", 0.1)),
                lambda_contact=float(wcfg.get("lambda_contact", 0.1)),
            ),
            oracle=OsimOracleConfig(
                time_reduce=str(ocfg.get("time_reduce", "mean")),
                robust=str(ocfg.get("robust", "huber")),
                huber_delta=float(ocfg.get("huber_delta", 10.0)),
                charbonnier_eps=float(ocfg.get("charbonnier_eps", 1e-3)),
                cvar_alpha=float(ocfg.get("cvar_alpha", 0.1)),
                lse_temperature=float(ocfg.get("lse_temperature", 10.0)),
                t_weight_schedule=str(ocfg.get("t_weight_schedule", "none")),
                use_oracle_numpy=bool(ocfg.get("use_oracle_numpy", True)),
                smooth_poses=bool(ocfg.get("smooth_poses", True)),
                smooth_cutoff_hz=float(ocfg.get("smooth_cutoff_hz", 6.0)),
                smooth_butterworth_order=int(ocfg.get("smooth_butterworth_order", 2)),
                opensim_model_path=(str(ocfg["opensim_model_path"]) if ocfg.get("opensim_model_path") else None),
                opensim_max_frames=int(ocfg.get("opensim_max_frames", 64)),
                moco_weld_toes=bool(ocfg.get("moco_weld_toes", True)),
                moco_markers_global_weight=float(ocfg.get("moco_markers_global_weight", 10.0)),
                moco_control_effort_weight=float(ocfg.get("moco_control_effort_weight", 0.1)),
                moco_mesh_interval=(
                    None if ocfg.get("moco_mesh_interval") is None else float(ocfg.get("moco_mesh_interval"))
                ),
                moco_markers_lowpass_hz=float(ocfg.get("moco_markers_lowpass_hz", 0.0)),
                moco_max_solver_iterations=int(ocfg.get("moco_max_solver_iterations", 200)),
                moco_convergence_tolerance=float(ocfg.get("moco_convergence_tolerance", 5e-2)),
                moco_constraint_tolerance=float(ocfg.get("moco_constraint_tolerance", 5e-2)),
            ),
        )

    print(
        f"[train] guidance={guidance_mode.value} device={device} data_root={data_root} cwd={os.getcwd()}",
        flush=True,
    )
    if guidance_mode == GuidanceMode.SINDY:
        print(f"[train] lambda_sindy={lambda_sindy} sindy_dir={sindy_dir!r}", flush=True)
    if guidance_mode == GuidanceMode.OSIM:
        print(
            f"[train] lambda_osim={lambda_osim} "
            f"osim.time_reduce={osim_guidance.oracle.time_reduce if osim_guidance else 'n/a'} "
            f"osim.robust={osim_guidance.oracle.robust if osim_guidance else 'n/a'} "
            f"osim.t_weight={osim_guidance.oracle.t_weight_schedule if osim_guidance else 'n/a'} "
            f"osim.numpy_oracle={osim_guidance.oracle.use_oracle_numpy if osim_guidance else 'n/a'} "
            f"osim.oracle_physics=moco_track",
            flush=True,
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(str(out / "config_resolved.json"), cfg)

    max_steps = int(train_cfg.get("max_steps", 100000))
    log_every = int(train_cfg.get("log_every", 100))
    save_every = int(train_cfg.get("save_every", 5000))
    cond_drop_prob = float(train_cfg.get("cond_drop_prob", 0.1))

    train_iter = iter(train_loader)
    for step in range(1, max_steps + 1):
        try:
            motion, captions, _sample_ids = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            motion, captions, _sample_ids = next(train_iter)
        x0 = motion.to(device)
        text_in = list(captions)
        if cond_drop_prob > 0:
            drop_mask = torch.rand((len(text_in),), device=device) < cond_drop_prob
            for i in range(len(text_in)):
                if bool(drop_mask[i].item()):
                    text_in[i] = ""
        text_ctx, text_mask = _clip_text_ctx(clip_model, text_in, device=device)
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
            loss_guidance = float(lambda_sindy) * sindy_guidance.loss(x0_pred, captions=text_in, device=device)
        elif guidance_mode == GuidanceMode.OSIM and osim_guidance is not None:
            raw_guide, guide_stats = osim_guidance.loss_and_stats(x0_pred)
            t_weight = osim_guidance.guidance_weight(t=t, total_timesteps=int(sched.timesteps))
            loss_guidance = float(lambda_osim) * t_weight * raw_guide

        loss = loss_diff + loss_guidance
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg.get("grad_clip", 1.0)))
        opt.step()

        if step % log_every == 0:
            msg = (
                f"step={step} loss={float(loss.item()):.6f} "
                f"diff={float(loss_diff.item()):.6f} guide={float(loss_guidance.item()):.6f}"
            )
            if guidance_mode == GuidanceMode.OSIM and guide_stats:
                msg += (
                    f" osim_vel={guide_stats.get('osim_vel', 0.0):.4f}"
                    f" osim_acc={guide_stats.get('osim_acc', 0.0):.4f}"
                    f" osim_tau={guide_stats.get('osim_torque', 0.0):.4f}"
                    f" osim_jerk={guide_stats.get('osim_jerk', 0.0):.4f}"
                    f" osim_eff={guide_stats.get('osim_effort', 0.0):.4f}"
                    f" osim_contact_gap={guide_stats.get('osim_contact_gap', 0.0):.4f}"
                )
                if "osim_oracle_scalar" in guide_stats:
                    msg += (
                        f" oracle={guide_stats.get('osim_oracle_scalar', 0.0):.4f}"
                        f" oracle_tau={guide_stats.get('osim_oracle_torque', 0.0):.4f}"
                    )
            print(msg)
        if step % save_every == 0:
            torch.save({"model_state": model.state_dict(), "step": step, "feature_dim": int(train_ds.feature_dim)}, out / f"ckpt_step_{step:07d}.pt")

    torch.save({"model_state": model.state_dict(), "step": max_steps, "feature_dim": int(train_ds.feature_dim)}, out / "last.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train text-conditioned diffusion model with guidance modes.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    train(config_path=str(args.config), out_dir=str(args.out_dir))


if __name__ == "__main__":
    main()

