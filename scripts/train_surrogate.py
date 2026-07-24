"""Train q → muscle activation surrogate on cached B3D OpenSim labels."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from common.distributed import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    log_gpu_diagnostics,
    log_main,
    maybe_relaunch_with_torchrun,
    model_state_dict,
    parse_distributed_enabled,
    resolve_nproc_per_node,
    should_auto_relaunch_torchrun,
    resolve_train_device,
    seed_all,
    setup_spawn_if_distributed,
    wrap_ddp,
)
from common.paths import activation_surrogate_latest_link, default_humanml3d_root, update_latest_symlink
from common.run_setup import (
    default_config_path,
    require_nimble_b3d,
    require_nimble_normalization,
    resolve_run_dir,
    resolve_training_data_root,
)
from common.run_logging import RunLogger, add_run_log_cli_args, get_run_logger, run_logged_main
from surrogate.dataset import ActivationB3DDataset
from surrogate.model import build_activation_surrogate


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def activation_surrogate_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    lambda_temporal: float = 0.1,
) -> torch.Tensor:
    """L1 reconstruction + temporal L1 on frame-to-frame deltas."""
    loss_main = F.l1_loss(pred, target)
    if pred.shape[1] < 2 or float(lambda_temporal) <= 0.0:
        return loss_main
    dp = pred[:, 1:] - pred[:, :-1]
    dt = target[:, 1:] - target[:, :-1]
    loss_temporal = F.l1_loss(dp, dt)
    return loss_main + float(lambda_temporal) * loss_temporal


def _per_muscle_l1(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """Mean L1 per muscle channel over batch and time → ``[M]``."""
    err = torch.abs(pred - target).mean(dim=(0, 1))
    return err.detach().cpu().numpy().astype(np.float64)


def train_activation_surrogate(
    *,
    data_root: str,
    output: str,
    split: str = "train",
    val_split: str = "val",
    window_size: int = 64,
    window_stride: int = 16,
    normalize_q: bool = True,
    max_motions: int = 0,
    skip_zero_placeholders: bool = True,
    model_type: str = "mlp",
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.1,
    num_heads: int = 4,
    dim_feedforward: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 50,
    batch_size: int = 32,
    lambda_temporal: float = 0.1,
    device_name: str = "auto",
    num_workers: int = 0,
    seed: int = 42,
    distributed_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    use_ddp = init_distributed(distributed_cfg=distributed_cfg)
    seed_all(int(seed))
    data_root = resolve_training_data_root(data_root)
    require_nimble_b3d(data_root)
    require_nimble_normalization(data_root)
    device = resolve_train_device(device_name)
    if not use_ddp:
        log_gpu_diagnostics()

    train_ds = ActivationB3DDataset(
        data_root,
        split=split,
        window_size=window_size,
        window_stride=window_stride,
        normalize_q=normalize_q,
        max_motions=max_motions,
        skip_zero_placeholders=skip_zero_placeholders,
    )
    try:
        val_ds = ActivationB3DDataset(
            data_root,
            split=val_split,
            window_size=window_size,
            window_stride=window_stride,
            normalize_q=normalize_q,
            max_motions=max_motions,
            skip_zero_placeholders=skip_zero_placeholders,
        )
    except ValueError:
        val_ds = None

    logger = get_run_logger()
    logger.progress(
        f"train split={split}: windows={len(train_ds)} "
        f"motions_kept={train_ds.num_motions_kept} "
        f"skipped_zero={train_ds.num_motions_skipped_zero}"
    )
    if val_ds is not None:
        logger.progress(
            f"val split={val_split}: windows={len(val_ds)} "
            f"motions_kept={val_ds.num_motions_kept} "
            f"skipped_zero={val_ds.num_motions_skipped_zero}"
        )

    per_gpu_batch = int(batch_size)
    train_sampler = None
    val_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            drop_last=False,
        )
        if val_ds is not None:
            val_sampler = DistributedSampler(
                val_ds,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=False,
                drop_last=False,
            )
    train_loader = DataLoader(
        train_ds,
        batch_size=per_gpu_batch,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=per_gpu_batch,
            shuffle=False,
            sampler=val_sampler,
            num_workers=int(num_workers),
            pin_memory=device.type == "cuda",
        )
        if val_ds is not None
        else None
    )

    sample_q, sample_act = train_ds[0]
    ckpt_payload = {
        "model_type": str(model_type),
        "input_dim": int(sample_q.shape[-1]),
        "output_dim": int(sample_act.shape[-1]),
        "window_size": int(window_size),
        "normalize_q": bool(normalize_q),
        "skip_zero_placeholders": bool(skip_zero_placeholders),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "num_heads": int(num_heads),
        "dim_feedforward": int(dim_feedforward),
        "d_model": int(dim_feedforward) if str(model_type) == "transformer" else 0,
        "seed": int(seed),
    }
    model = build_activation_surrogate(
        model_type=model_type,
        input_dim=int(ckpt_payload["input_dim"]),
        output_dim=int(ckpt_payload["output_dim"]),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        max_seq_len=window_size,
    ).to(device)
    find_unused = bool((distributed_cfg or {}).get("find_unused_parameters", False))
    model = wrap_ddp(model, find_unused_parameters=find_unused)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )

    log_main(
        f"[activation_surrogate] device={device} world_size={get_world_size()} "
        f"distributed={use_ddp} per_gpu_batch={per_gpu_batch} "
        f"global_batch={per_gpu_batch * get_world_size()}"
    )

    out_dir = Path(output)
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "latest.pt"
    best_path = out_dir / "best.pt"
    best_val = float("inf")
    history: list[dict[str, float]] = []

    def _save_checkpoint(path: Path, *, epoch: int, val_loss: float) -> None:
        if not is_main_process():
            return
        torch.save(
            {
                **ckpt_payload,
                "model_state_dict": model_state_dict(model),
                "epoch": int(epoch),
                "val_loss": float(val_loss),
            },
            path,
        )

    for epoch in range(int(epochs)):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        n_batches = 0
        for q, act in train_loader:
            q = q.to(device)
            act = act.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(q)
            loss = activation_surrogate_loss(
                pred, act, lambda_temporal=lambda_temporal
            )
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            n_batches += 1
        train_loss /= max(1, n_batches)

        val_loss = float("nan")
        val_muscle_l1_mean = float("nan")
        val_muscle_l1_max = float("nan")
        if val_loader is not None and is_main_process():
            model.eval()
            vsum = 0.0
            vb = 0
            muscle_l1_accum: np.ndarray | None = None
            muscle_count = 0
            with torch.no_grad():
                for q, act in val_loader:
                    q = q.to(device)
                    act = act.to(device)
                    pred = model(q)
                    vsum += float(
                        activation_surrogate_loss(
                            pred, act, lambda_temporal=lambda_temporal
                        ).item()
                    )
                    per_m = _per_muscle_l1(pred, act)
                    if muscle_l1_accum is None:
                        muscle_l1_accum = np.zeros_like(per_m)
                    muscle_l1_accum += per_m
                    muscle_count += 1
                    vb += 1
            val_loss = vsum / max(1, vb)
            if muscle_l1_accum is not None and muscle_count > 0:
                muscle_l1_accum /= float(muscle_count)
                val_muscle_l1_mean = float(np.mean(muscle_l1_accum))
                val_muscle_l1_max = float(np.max(muscle_l1_accum))
            if val_loss < best_val:
                best_val = val_loss
                _save_checkpoint(best_path, epoch=epoch, val_loss=val_loss)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_muscle_l1_mean": val_muscle_l1_mean,
                "val_muscle_l1_max": val_muscle_l1_max,
            }
        )
        if is_main_process():
            logger.progress(
                f"epoch {epoch + 1}/{epochs} train={train_loss:.6f} val={val_loss:.6f} "
                f"val_muscle_l1_mean={val_muscle_l1_mean:.6f} val_muscle_l1_max={val_muscle_l1_max:.6f}"
            )
            logger.verbose(
                f"[activation_surrogate] epoch {epoch + 1}/{epochs} "
                f"train={train_loss:.6f} val={val_loss:.6f}"
            )

    if is_main_process():
        _save_checkpoint(
            latest_path,
            epoch=int(epochs) - 1,
            val_loss=history[-1]["val_loss"] if history else float("nan"),
        )
        if not best_path.is_file():
            _save_checkpoint(
                best_path,
                epoch=int(epochs) - 1,
                val_loss=history[-1]["val_loss"] if history else float("nan"),
            )

    if not is_main_process():
        return {
            "checkpoint": str(latest_path),
            "best_checkpoint": str(best_path),
            "train_windows": len(train_ds),
            "val_windows": len(val_ds) if val_ds is not None else 0,
            "best_val_loss": best_val,
        }

    metrics = {
        "checkpoint": str(latest_path),
        "best_checkpoint": str(best_path),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds) if val_ds is not None else 0,
        "train_motions_kept": int(train_ds.num_motions_kept),
        "train_motions_skipped_zero": int(train_ds.num_motions_skipped_zero),
        "val_motions_kept": int(val_ds.num_motions_kept) if val_ds is not None else 0,
        "val_motions_skipped_zero": (
            int(val_ds.num_motions_skipped_zero) if val_ds is not None else 0
        ),
        "best_val_loss": best_val,
        "history": history,
    }
    (out_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    update_latest_symlink(run_dir=out_dir, latest_link=activation_surrogate_latest_link())
    return metrics


def _apply_json_config(args: argparse.Namespace, config_path: str) -> None:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing config: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key, value in cfg.items():
        if key in {"output", "distributed", "seed"}:
            continue
        if hasattr(args, key):
            setattr(args, key, value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train muscle activation surrogate")
    parser.add_argument(
        "--config",
        default="",
        help="JSON config (default: configs/train_surrogate.json)",
    )
    parser.add_argument("--data_root", default=default_humanml3d_root())
    parser.add_argument("--output", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--window_stride", type=int, default=16)
    parser.add_argument("--normalize_q", type=int, default=1)
    parser.add_argument("--max_motions", type=int, default=0)
    parser.add_argument(
        "--skip_zero_placeholders",
        type=int,
        default=1,
        help="1=drop motions whose muscle_activations are all-zero fallbacks",
    )
    parser.add_argument("--model_type", default="mlp")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lambda_temporal", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    add_run_log_cli_args(parser)
    args = parser.parse_args()

    cfg_path = str(args.config).strip()
    dist_cfg: Dict[str, Any] = {}
    full_cfg: Dict[str, Any] = {}
    if not cfg_path:
        default_cfg = default_config_path("train_surrogate.json")
        if default_cfg.is_file():
            cfg_path = str(default_cfg)
    if cfg_path:
        _apply_json_config(args, cfg_path)
        full_cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        if isinstance(full_cfg.get("distributed"), dict):
            dist_cfg = full_cfg["distributed"]

    args.data_root = resolve_training_data_root(args.data_root)
    out_override = os.environ.get("ACTIVATION_SURROGATE_OUTPUT", "").strip()
    args.output = str(
        resolve_run_dir(
            str(args.output).strip() or out_override or None,
            family="activation_surrogate",
        )
    )

    if should_auto_relaunch_torchrun(dist_cfg):
        maybe_relaunch_with_torchrun()
    setup_spawn_if_distributed()

    def _run(_logger: RunLogger) -> None:
        try:
            metrics = train_activation_surrogate(
                data_root=args.data_root,
                output=args.output,
                split=args.split,
                val_split=args.val_split,
                window_size=int(args.window_size),
                window_stride=int(args.window_stride),
                normalize_q=bool(int(args.normalize_q)),
                max_motions=int(args.max_motions),
                skip_zero_placeholders=bool(int(args.skip_zero_placeholders)),
                model_type=args.model_type,
                hidden_dim=int(args.hidden_dim),
                num_layers=int(args.num_layers),
                dropout=float(args.dropout),
                num_heads=int(args.num_heads),
                dim_feedforward=int(args.dim_feedforward),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lambda_temporal=float(args.lambda_temporal),
                device_name=args.device,
                num_workers=int(args.num_workers),
                seed=int(full_cfg.get("seed", args.seed)) if cfg_path else int(args.seed),
                distributed_cfg=dist_cfg,
            )
            if is_main_process():
                _logger.verbose(json.dumps(metrics, indent=2))
        finally:
            cleanup_distributed()
        sys.exit(0)

    run_logged_main(
        Path(__file__).stem,
        args.log_dir,
        _run,
        argv=sys.argv,
        no_run_log=bool(args.no_run_log),
    )


if __name__ == "__main__":
    main()
