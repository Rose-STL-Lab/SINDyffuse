"""Train text→Xi SINDy model on MinT L_bio + muscle activation targets."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common.clip_model import load_clip
import joblib
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from common.distributed import (
    barrier,
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_process,
    log_main,
    log_gpu_diagnostics,
    maybe_relaunch_with_torchrun,
    model_state_dict,
    parse_distributed_enabled,
    resolve_nproc_per_node,
    should_auto_relaunch_torchrun,
    resolve_train_device,
    seed_all,
    setup_spawn_if_distributed,
    shard_indices,
    unwrap_module as unwrap_train_module,
    wrap_ddp,
)
from common.paths import (
    default_humanml3d_root,
    humanml3d_text_dir,
    sindy_latest_link,
    update_latest_symlink,
)
from common.run_setup import default_config_path, require_motion_cache, resolve_run_dir, resolve_training_data_root
from common.run_logging import RunLogger, add_run_log_cli_args, get_run_logger, run_logged_main
from common.biomech import BIOMECH_COMPONENT_KEYS

from sindy.dataset import SindyWindowDataset, prepare_lazy_sindy_data
from sindy.targets import (
    N_BIO_TARGETS,
    N_MUSCLE_TARGETS,
    N_SINDY_TARGETS,
    muscle_channel_names,
    parse_target_weights,
    sindy_target_keys,
)
from sindy.windows import collect_windows, make_theta_spec
from sindy.library import ThetaLibrary
from sindy.model import TextToXi, predict_from_xi, xi_temporal_smoothness


def _build_lr_scheduler(
    opt: optim.Optimizer,
    *,
    scheduler_name: str,
    epochs: int,
    warmup_epochs: int,
    lr_min: float,
):
    name = str(scheduler_name).strip().lower()
    if name in {"", "none"}:
        return None
    total = max(1, int(epochs))
    warmup = max(0, min(int(warmup_epochs), total - 1))
    eta_min = float(lr_min)
    base_lr = float(opt.param_groups[0]["lr"])
    if name == "cosine":
        eta_min_ratio = eta_min / max(base_lr, 1e-12)

        def lr_lambda(epoch_idx: int) -> float:
            if warmup > 0 and epoch_idx < warmup:
                return 0.01 + (1.0 - 0.01) * (float(epoch_idx) / float(warmup))
            if total <= warmup:
                return eta_min_ratio
            progress = (float(epoch_idx) - float(warmup)) / float(max(1, total - warmup))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

        return optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    if name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=max(5, total // 20),
            min_lr=eta_min,
        )
    raise ValueError(f"Unknown scheduler: {scheduler_name!r} (expected cosine, plateau, or none)")


def _resolve_device(name: str) -> torch.device:
    n = str(name).strip().lower()
    if n == "cpu":
        return torch.device("cpu")
    if n == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_caption_segments(text_path: Path, fps: float) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    if not text_path.is_file():
        return out
    for line in text_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("#")
        if len(parts) != 4:
            continue
        caption = parts[0].strip()
        if not caption:
            continue
        try:
            f_tag, to_tag = float(parts[2]), float(parts[3])
        except Exception:
            f_tag, to_tag = 0.0, 0.0
        if np.isnan(f_tag):
            f_tag = 0.0
        if np.isnan(to_tag):
            to_tag = 0.0
        if f_tag == 0.0 and to_tag == 0.0:
            out.append((caption, -1, -1))
        else:
            st, ed = int(max(0.0, f_tag) * fps), int(max(0.0, to_tag) * fps)
            if ed > st:
                out.append((caption, st, ed))
    return out


def _caption_for_window(
    sample_id: str,
    text_dir: Path,
    fps: float,
    window_size: int,
    cache: Dict[str, List[Tuple[str, int, int]]],
) -> str:
    if ":start" in sample_id:
        sid, st_txt = sample_id.split(":start", 1)
        start = int(st_txt)
    else:
        sid, start = sample_id, 0
    if sid not in cache:
        cache[sid] = _load_caption_segments(text_dir / f"{sid}.txt", fps=fps)
    segments = cache[sid]
    if not segments:
        return "a person moves"
    end = start + int(window_size)
    best_caption, best_overlap = None, -1
    full_caption = None
    for caption, st, ed in segments:
        if st < 0 and ed < 0:
            full_caption = caption
            continue
        overlap = max(0, min(end, ed) - max(start, st))
        if overlap > best_overlap:
            best_overlap, best_caption = overlap, caption
    if best_caption is not None and best_overlap > 0:
        return best_caption
    return full_caption if full_caption is not None else segments[0][0]


@torch.no_grad()
def _text_embed(
    captions: List[str],
    device: torch.device,
    model_name: str = "ViT-B/32",
    batch_size: int = 128,
) -> np.ndarray:
    model, _ = load_clip(model_name, device=device, jit=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    parts: List[np.ndarray] = []
    bs = max(1, int(batch_size))
    for i in range(0, len(captions), bs):
        toks = clip.tokenize(captions[i : i + bs], truncate=True).to(device)
        parts.append(model.encode_text(toks).float().cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32) if parts else np.zeros((0, 512), np.float32)


def _run_training_loop(
    *,
    model: TextToXi,
    opt: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    length: int,
    tr_idx: np.ndarray,
    val_idx: np.ndarray,
    captions: List[str],
    cap_to_idx: Dict[str, int],
    emb_t: torch.Tensor,
    epochs: int,
    batch_size: int,
    lambda_l1: float,
    lambda_xi_smooth: float,
    grad_clip: float,
    early_stop_patience: int,
    rng: np.random.Generator,
    fetch_batch: Any,
    run_validation: bool = True,
    n_bio_targets: int = N_BIO_TARGETS,
    target_weights: torch.Tensor | None = None,
) -> Tuple[float, Optional[Dict[str, torch.Tensor]], Dict[str, float], int]:
    """Shared epoch loop for preloaded tensors or lazy dataset batches."""
    best_val, best_state = float("inf"), None
    best_metrics: Dict[str, float] = {}
    w = target_weights
    stale_epochs = 0
    last_epoch = 0
    plateau_sched = isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)
    for ep in range(1, int(epochs) + 1):
        last_epoch = ep
        model.train()
        rng.shuffle(tr_idx)
        losses = []
        for st in range(0, len(tr_idx), int(batch_size)):
            bidx = tr_idx[st : st + int(batch_size)]
            if len(bidx) == 0:
                continue
            theta_b, y_b = fetch_batch(bidx)
            loss_fit = loss_sparse = loss_smooth = torch.tensor(0.0, device=device)
            for bi, sample_idx in enumerate(bidx):
                cap = captions[int(sample_idx)]
                e = emb_t[cap_to_idx[cap] : cap_to_idx[cap] + 1]
                xi_out = model(e, seq_len=int(length))
                pred = predict_from_xi(theta_b[bi : bi + 1], xi_out)
                err = (pred - y_b[bi : bi + 1]) ** 2
                if w is not None:
                    err = err * w.view(1, 1, -1)
                loss_fit = loss_fit + torch.mean(err)
                xi_tensor = xi_out[0] if isinstance(xi_out, tuple) else xi_out
                loss_sparse = loss_sparse + torch.mean(torch.abs(xi_tensor))
                loss_smooth = loss_smooth + xi_temporal_smoothness(xi_out)
            denom = float(max(1, len(bidx)))
            loss = loss_fit / denom + float(lambda_l1) * (loss_sparse / denom) + float(lambda_xi_smooth) * (loss_smooth / denom)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_loss = float("inf")
        val_bio = float("nan")
        val_muscle = float("nan")
        if run_validation and len(val_idx) > 0:
            model.eval()
            val_loss = val_bio_sum = val_muscle_sum = 0.0
            with torch.no_grad():
                for sample_idx in val_idx:
                    cap = captions[int(sample_idx)]
                    e = emb_t[cap_to_idx[cap] : cap_to_idx[cap] + 1]
                    theta_b, y_b = fetch_batch([int(sample_idx)])
                    xi_out = model(e, seq_len=int(length))
                    pred = predict_from_xi(theta_b, xi_out)
                    err = (pred - y_b) ** 2
                    if w is not None:
                        err = err * w.view(1, 1, -1)
                    val_loss += float(torch.mean(err).cpu())
                    val_bio_sum += float(torch.mean(err[..., :n_bio_targets]).cpu())
                    val_muscle_sum += float(torch.mean(err[..., n_bio_targets:]).cpu())
            n_val = max(1, len(val_idx))
            val_loss /= n_val
            val_bio = val_bio_sum / n_val
            val_muscle = val_muscle_sum / n_val
            if val_loss < best_val:
                best_val = val_loss
                best_metrics = {"val_bio_mse": val_bio, "val_muscle_mse": val_muscle}
                best_state = {k: v.detach().cpu().clone() for k, v in model_state_dict(model).items()}
                stale_epochs = 0
            else:
                stale_epochs += 1

        if scheduler is not None:
            if plateau_sched:
                if run_validation and len(val_idx) > 0:
                    scheduler.step(val_loss)
            else:
                scheduler.step()
            if plateau_sched and is_distributed():
                lr_t = torch.tensor([float(opt.param_groups[0]["lr"])], device=device)
                torch.distributed.broadcast(lr_t, src=0)
                for pg in opt.param_groups:
                    pg["lr"] = float(lr_t.item())

        should_stop = 0
        if (
            is_main_process()
            and int(early_stop_patience) > 0
            and len(val_idx) > 0
            and stale_epochs >= int(early_stop_patience)
        ):
            should_stop = 1
            from common.run_logging import get_run_logger

            get_run_logger().progress(
                f"early_stop at epoch={ep} (patience={int(early_stop_patience)}, best_val_loss={best_val:.6f})"
            )
        if is_distributed():
            stop_flag = torch.tensor([should_stop], device=device, dtype=torch.int32)
            torch.distributed.broadcast(stop_flag, src=0)
            should_stop = int(stop_flag.item())
        if should_stop:
            break

        if (ep % max(1, epochs // 20) == 0 or ep == 1) and is_main_process():
            vl = val_loss if run_validation else float("nan")
            from common.run_logging import get_run_logger

            lr_now = float(opt.param_groups[0]["lr"])
            get_run_logger().progress(
                f"epoch={ep}/{epochs} lr={lr_now:.2e} train_loss={np.mean(losses):.6f} val_loss={vl:.6f} "
                f"val_bio_mse={val_bio:.6f} val_muscle_mse={val_muscle:.6f}"
            )
    best_metrics["epochs_run"] = float(last_epoch)
    return best_val, best_state, best_metrics


def train(
    *,
    data_root: str,
    split: str,
    fps: float,
    output: str,
    window_size: int,
    window_stride: int,
    theta_tier: str,
    include_u: bool,
    include_c: bool,
    max_samples: int,
    lr: float,
    lr_min: float,
    warmup_epochs: int,
    scheduler_name: str,
    weight_decay: float,
    grad_clip: float,
    early_stop_patience: int,
    epochs: int,
    batch_size: int,
    lambda_l1: float,
    lambda_xi_smooth: float,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    ff_mult: int,
    num_experts: int,
    fallback_caption_weight: float,
    clip_model_name: str,
    clip_text_batch_size: int,
    device_name: str,
    bio_log_every: int,
    preload: bool = False,
    skip_zero_placeholders: bool = True,
    zero_atol: float = 1e-8,
    target_weights: Sequence[float] | None = None,
    distributed_cfg: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    use_ddp = init_distributed(distributed_cfg=distributed_cfg)
    if not use_ddp:
        log_gpu_diagnostics()
    seed_all(int(seed))
    data_root = resolve_training_data_root(data_root)
    from common.run_setup import require_motion_cache

    require_motion_cache(data_root)
    root = Path(data_root)
    bio_keys = list(BIOMECH_COMPONENT_KEYS)
    muscle_keys = list(muscle_channel_names())
    target_keys = list(sindy_target_keys())
    length = int(window_size) - 1
    u_names: List[str] = []
    feature_names: List[str] = []
    target_dim = 0
    f = 0
    theta_scaler: StandardScaler
    y_scaler: StandardScaler
    lazy_dataset: SindyWindowDataset | None = None

    if preload:
        log_main("[sindy/train] preload=True: loading all windows into memory")
        u, c, y, u_names, c_names, sample_ids = collect_windows(
            data_root,
            split,
            fps,
            window_size,
            window_stride,
            max_samples,
            log_every=bio_log_every,
            skip_zero_placeholders=skip_zero_placeholders,
            zero_atol=zero_atol,
        )
        n, t, _u_dim = u.shape
        u_in = u[:, :-1, :] if include_u else None
        c_in = c[:, :-1, :] if include_c else None
        spec = make_theta_spec(theta_tier, include_u, include_c, u_names)
        theta_flat, feature_names = ThetaLibrary(spec=spec).build(
            u=u_in,
            c=c_in,
            u_names=u_names if include_u else [],
            c_names=c_names if include_c else [],
        )
        length = t - 1
        f = int(theta_flat.shape[1])
        theta = theta_flat.reshape(n, length, f).astype(np.float32)
        if y.shape[1] != length:
            raise ValueError(f"y length {y.shape[1]} != theta length {length}")
        if y.shape[2] != N_SINDY_TARGETS:
            raise ValueError(f"Expected y target dim {N_SINDY_TARGETS}, got {y.shape[2]}")
        target_dim = int(y.shape[2])
        theta_scaler = StandardScaler()
        y_scaler = StandardScaler()
        theta_s = theta_scaler.fit_transform(theta.reshape(-1, f)).reshape(n, length, f).astype(np.float32)
        y_s = y_scaler.fit_transform(y.reshape(-1, target_dim)).reshape(n, length, target_dim).astype(np.float32)
    else:
        log_main("[sindy/train] preload=False: indexing windows and fitting scalers (on-demand reads)")
        index, theta_scaler, y_scaler, u_names, c_names, feature_names, target_dim, spec = prepare_lazy_sindy_data(
            data_root,
            split,
            fps=fps,
            window_size=window_size,
            window_stride=window_stride,
            max_samples=max_samples,
            theta_tier=theta_tier,
            include_u=include_u,
            include_c=include_c,
            log_every=bio_log_every,
            skip_zero_placeholders=skip_zero_placeholders,
            zero_atol=zero_atol,
        )
        if target_dim != N_SINDY_TARGETS:
            raise ValueError(f"Expected target_dim {N_SINDY_TARGETS}, got {target_dim}")
        f = len(feature_names)
        sample_ids = np.array([e.sample_id for e in index.entries], dtype=object)
        n = len(sample_ids)
        lazy_dataset = SindyWindowDataset(
            data_root,
            index,
            window_size=window_size,
            fps=fps,
            theta_spec=spec,
            include_u=include_u,
            include_c=include_c,
            theta_scaler=theta_scaler,
            y_scaler=y_scaler,
            u_names=u_names,
            c_names=c_names,
            feature_names=feature_names,
            target_dim=target_dim,
        )
        theta_s = None
        y_s = None

    text_dir = humanml3d_text_dir(data_root)
    caption_cache: Dict[str, List[Tuple[str, int, int]]] = {}
    captions = [
        _caption_for_window(str(sid), text_dir, fps, window_size, caption_cache) for sid in sample_ids.tolist()
    ]

    device = resolve_train_device(device_name)
    unique_caps = sorted(set(captions))
    unique_emb = _text_embed(unique_caps, device=device, model_name=clip_model_name, batch_size=clip_text_batch_size)
    emb_scaler = StandardScaler()
    unique_emb = emb_scaler.fit_transform(unique_emb).astype(np.float32)
    cap_to_idx = {c: i for i, c in enumerate(unique_caps)}

    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    val_n = max(1, int(0.1 * n))
    val_idx_full, tr_idx_full = idx[:val_n], idx[val_n:] if len(idx[val_n:]) > 0 else idx[:val_n]
    tr_idx = shard_indices(tr_idx_full)
    val_idx = val_idx_full if is_main_process() else np.array([], dtype=np.int64)

    find_unused = bool((distributed_cfg or {}).get("find_unused_parameters", False))
    model = TextToXi(
        in_dim=unique_emb.shape[1],
        hidden_dim=int(hidden_dim),
        theta_dim=f,
        target_dim=target_dim,
        num_experts=int(num_experts),
        max_seq_len=int(length),
        num_layers=int(num_layers),
        dropout=float(dropout),
        ff_mult=int(ff_mult),
    ).to(device)
    model = wrap_ddp(model, find_unused_parameters=find_unused)
    opt = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scheduler = _build_lr_scheduler(
        opt,
        scheduler_name=str(scheduler_name),
        epochs=int(epochs),
        warmup_epochs=int(warmup_epochs),
        lr_min=float(lr_min),
    )
    log_main(
        f"[sindy/train] device={device} world_size={get_world_size()} "
        f"distributed={use_ddp} train_windows={len(tr_idx)} per_gpu_batch={batch_size} "
        f"target_dim={target_dim} (bio={N_BIO_TARGETS} muscle={N_MUSCLE_TARGETS}) "
        f"layers={num_layers} dropout={dropout} "
        f"scheduler={scheduler_name} weight_decay={weight_decay}"
    )
    emb_t = torch.tensor(unique_emb, dtype=torch.float32, device=device)
    tw_np = parse_target_weights(target_weights, n_targets=target_dim)
    target_w = torch.tensor(tw_np, dtype=torch.float32, device=device)

    if preload:
        assert theta_s is not None and y_s is not None
        theta_t = torch.tensor(theta_s, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_s, dtype=torch.float32, device=device)

        def fetch_batch(indices: List[int] | np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
            bidx_t = torch.tensor(indices, dtype=torch.long, device=device)
            return theta_t[bidx_t], y_t[bidx_t]
    else:
        assert lazy_dataset is not None

        def fetch_batch(indices: List[int] | np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
            items = [lazy_dataset[int(i)] for i in indices]
            theta_b = torch.stack([it["theta"] for it in items], dim=0).to(device)
            y_b = torch.stack([it["y"] for it in items], dim=0).to(device)
            return theta_b, y_b

    best_val, best_state, best_val_metrics = _run_training_loop(
        model=model,
        opt=opt,
        scheduler=scheduler,
        device=device,
        length=length,
        tr_idx=tr_idx,
        val_idx=val_idx,
        captions=captions,
        cap_to_idx=cap_to_idx,
        emb_t=emb_t,
        epochs=epochs,
        batch_size=batch_size,
        lambda_l1=lambda_l1,
        lambda_xi_smooth=lambda_xi_smooth,
        grad_clip=grad_clip,
        early_stop_patience=early_stop_patience,
        rng=rng,
        fetch_batch=fetch_batch,
        run_validation=is_main_process(),
        n_bio_targets=N_BIO_TARGETS,
        target_weights=target_w,
    )

    if best_state is not None:
        unwrap_train_module(model).load_state_dict(best_state)

    out = Path(output)
    if is_main_process():
        out.mkdir(parents=True, exist_ok=True)
    barrier()
    if not is_main_process():
        return {
            "config": {"distributed_rank": int(get_world_size())},
            "metrics": {"best_val_loss": float(best_val)},
        }

    torch.save(
        {
            "model_state": model_state_dict(model),
            "in_dim": int(unique_emb.shape[1]),
            "hidden_dim": int(hidden_dim),
            "theta_dim": f,
            "target_dim": target_dim,
            "num_experts": int(num_experts),
            "max_seq_len": int(length),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "ff_mult": int(ff_mult),
            "bio_channel_names": bio_keys,
            "muscle_channel_names": muscle_keys,
            "target_channel_names": target_keys,
        },
        out / "text_to_xi.pt",
    )
    joblib.dump(theta_scaler, out / "scaler_theta.pkl")
    joblib.dump(y_scaler, out / "scaler_y.pkl")
    joblib.dump(emb_scaler, out / "scaler_text_embed.pkl")

    results = {
        "config": {
            "data_root": data_root,
            "split": split,
            "fps": float(fps),
            "target_dim": target_dim,
            "bio_channel_names": bio_keys,
            "muscle_channel_names": muscle_keys,
            "target_channel_names": target_keys,
            "window_size": int(window_size),
            "window_stride": int(window_stride),
            "theta_tier": theta_tier,
            "preload": bool(preload),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "ff_mult": int(ff_mult),
            "lr": float(lr),
            "lr_min": float(lr_min),
            "warmup_epochs": int(warmup_epochs),
            "scheduler": str(scheduler_name),
            "weight_decay": float(weight_decay),
            "grad_clip": float(grad_clip),
            "early_stop_patience": int(early_stop_patience),
            "lambda_l1": float(lambda_l1),
            "lambda_xi_smooth": float(lambda_xi_smooth),
        },
        "metrics": {"best_val_loss": float(best_val), **best_val_metrics},
    }
    with open(out / "sild_text_train_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    update_latest_symlink(run_dir=out, latest_link=sindy_latest_link())
    return results


def _apply_json_config(args: argparse.Namespace, config_path: str) -> None:
    """Load ``train_sindy.json``; CLI ``--output`` always wins."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing config: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key, value in cfg.items():
        if key in {"output", "distributed", "seed"}:
            continue
        if key == "preload":
            setattr(args, "preload", bool(value))
            continue
        if hasattr(args, key):
            setattr(args, key, value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SINDy text→Xi on MinT L_bio targets")
    parser.add_argument(
        "--config",
        default="",
        help="Path to train_sindy.json (default: configs/train_sindy.json next to repo root)",
    )
    parser.add_argument("--data_root", default=default_humanml3d_root())
    parser.add_argument("--split", default="train")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--output", default="", help="Run directory (default: results/sindy/runs/<timestamp>)")
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--window_stride", type=int, default=16)
    parser.add_argument("--theta_tier", default="tier2_moderate")
    parser.add_argument("--include_u", type=int, default=1)
    parser.add_argument("--include_c", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--scheduler", default="cosine")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lambda_l1", type=float, default=1e-4)
    parser.add_argument("--lambda_xi_smooth", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--skip_zero_placeholders", type=int, default=1)
    parser.add_argument("--zero_atol", type=float, default=1e-8)
    parser.add_argument("--num_experts", type=int, default=1)
    parser.add_argument("--fallback_caption_weight", type=float, default=0.2)
    parser.add_argument("--clip_model_name", default="ViT-B/32")
    parser.add_argument("--clip_text_batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bio_log_every", type=int, default=50)
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load all NPZ windows into RAM before training (default: read on demand)",
    )
    add_run_log_cli_args(parser)
    args = parser.parse_args()
    cfg_path = str(args.config).strip()
    dist_cfg: Dict[str, Any] = {}
    full_cfg: Dict[str, Any] = {}
    if not cfg_path:
        default_cfg = default_config_path("train_sindy.json")
        if default_cfg.is_file():
            cfg_path = str(default_cfg)
    if cfg_path:
        _apply_json_config(args, cfg_path)
        full_cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        if isinstance(full_cfg.get("distributed"), dict):
            dist_cfg = full_cfg["distributed"]
    else:
        full_cfg = {}

    args.data_root = resolve_training_data_root(args.data_root)
    args.output = str(
        resolve_run_dir(str(args.output).strip() or None, family="sindy")
    )

    if should_auto_relaunch_torchrun(dist_cfg):
        maybe_relaunch_with_torchrun(module="sindy.train")
    setup_spawn_if_distributed()

    def _run(_logger: RunLogger) -> None:
        try:
            metrics = train(
                data_root=args.data_root,
                split=args.split,
                fps=float(args.fps),
                output=args.output,
                window_size=int(args.window_size),
                window_stride=int(args.window_stride),
                theta_tier=args.theta_tier,
                include_u=bool(args.include_u),
                include_c=bool(args.include_c),
                max_samples=int(args.max_samples),
                lr=float(args.lr),
                lr_min=float(args.lr_min),
                warmup_epochs=int(args.warmup_epochs),
                scheduler_name=str(args.scheduler),
                weight_decay=float(args.weight_decay),
                grad_clip=float(args.grad_clip),
                early_stop_patience=int(args.early_stop_patience),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lambda_l1=float(args.lambda_l1),
                lambda_xi_smooth=float(args.lambda_xi_smooth),
                hidden_dim=int(args.hidden_dim),
                num_layers=int(args.num_layers),
                dropout=float(args.dropout),
                ff_mult=int(args.ff_mult),
                num_experts=int(args.num_experts),
                fallback_caption_weight=float(args.fallback_caption_weight),
                clip_model_name=args.clip_model_name,
                clip_text_batch_size=int(args.clip_text_batch_size),
                device_name=args.device,
                bio_log_every=int(args.bio_log_every),
                preload=bool(args.preload),
                skip_zero_placeholders=bool(args.skip_zero_placeholders),
                zero_atol=float(args.zero_atol),
                target_weights=full_cfg.get("target_weights") if cfg_path else None,
                distributed_cfg=dist_cfg,
                seed=int(full_cfg.get("seed", 42)) if cfg_path else 42,
            )
            if is_main_process():
                _logger.verbose(json.dumps(metrics, indent=2))
        finally:
            cleanup_distributed()
        sys.exit(0)

    rank = get_rank() if is_distributed() else None
    if args.no_run_log:
        _run(RunLogger(terminal=__import__("sys").stdout, log_file=None))
    else:
        from common.run_logging import run_log_session

        script_name = Path(__file__).stem
        with run_log_session(
            args.log_dir,
            script_name=script_name,
            argv=sys.argv,
            rank=rank,
        ) as (_paths, logger):
            if is_main_process():
                logger.progress(f"log: {_paths.latest_log}")
            _run(logger)


if __name__ == "__main__":
    main()
