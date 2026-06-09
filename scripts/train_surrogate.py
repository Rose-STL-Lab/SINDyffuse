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
from datetime import datetime
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.paths import default_humanml3d_root
from common.run_logging import RunLogger, add_run_log_cli_args, get_run_logger, run_logged_main
from surrogate.dataset import ActivationB3DDataset
from surrogate.model import build_activation_surrogate


def activation_surrogate_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    lambda_temporal: float = 0.1,
) -> torch.Tensor:
    """Masked L1 reconstruction + temporal L1 on frame-to-frame deltas.

    ``mask`` is ``[B, T]`` with 1.0 on frames with valid MocoTrack labels.
    """
    m = mask.unsqueeze(-1)
    denom = m.sum().clamp_min(1.0)
    loss_main = (torch.abs(pred - target) * m).sum() / denom
    if pred.shape[1] < 2 or float(lambda_temporal) <= 0.0:
        return loss_main
    pair_mask = (mask[:, 1:] > 0.5) & (mask[:, :-1] > 0.5)
    if not pair_mask.any():
        return loss_main
    dp = pred[:, 1:] - pred[:, :-1]
    dt = target[:, 1:] - target[:, :-1]
    pm = pair_mask.unsqueeze(-1)
    denom_t = pm.sum().clamp_min(1.0)
    loss_temporal = (torch.abs(dp - dt) * pm).sum() / denom_t
    return loss_main + float(lambda_temporal) * loss_temporal


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
    min_window_valid_fraction: float = 0.5,
    require_activation_mask: bool = False,
    device_name: str = "auto",
    num_workers: int = 0,
) -> Dict[str, Any]:
    if str(device_name).strip().lower() == "cpu":
        device = torch.device("cpu")
    elif str(device_name).strip().lower() == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ActivationB3DDataset(
        data_root,
        split=split,
        window_size=window_size,
        window_stride=window_stride,
        normalize_q=normalize_q,
        max_motions=max_motions,
        min_window_valid_fraction=min_window_valid_fraction,
        require_activation_mask=require_activation_mask,
    )
    try:
        val_ds = ActivationB3DDataset(
            data_root,
            split=val_split,
            window_size=window_size,
            window_stride=window_stride,
            normalize_q=normalize_q,
            max_motions=max_motions,
            min_window_valid_fraction=min_window_valid_fraction,
            require_activation_mask=require_activation_mask,
        )
    except ValueError:
        val_ds = None

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers))
        if val_ds is not None
        else None
    )

    sample_q, _, _ = train_ds[0]
    model = build_activation_surrogate(
        model_type=model_type,
        input_dim=int(sample_q.shape[-1]),
        output_dim=int(train_ds[0][1].shape[-1]),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        max_seq_len=window_size,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "latest.pt"
    best_val = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(int(epochs)):
        model.train()
        train_loss = 0.0
        train_mask_frac = 0.0
        n_batches = 0
        for q, act, mask in train_loader:
            q = q.to(device)
            act = act.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(q)
            loss = activation_surrogate_loss(
                pred, act, mask, lambda_temporal=lambda_temporal
            )
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            train_mask_frac += float(mask.mean().item())
            n_batches += 1
        train_loss /= max(1, n_batches)
        train_mask_frac /= max(1, n_batches)

        val_loss = float("nan")
        val_mask_frac = float("nan")
        if val_loader is not None:
            model.eval()
            vsum = 0.0
            vmask = 0.0
            vb = 0
            with torch.no_grad():
                for q, act, mask in val_loader:
                    q = q.to(device)
                    act = act.to(device)
                    mask = mask.to(device)
                    pred = model(q)
                    vsum += float(
                        activation_surrogate_loss(
                            pred, act, mask, lambda_temporal=lambda_temporal
                        ).item()
                    )
                    vmask += float(mask.mean().item())
                    vb += 1
            val_loss = vsum / max(1, vb)
            val_mask_frac = vmask / max(1, vb)
            if val_loss < best_val:
                best_val = val_loss

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_mask_fraction": train_mask_frac,
                "val_mask_fraction": val_mask_frac,
            }
        )
        get_run_logger().progress(
            f"epoch {epoch + 1}/{epochs} train={train_loss:.6f} val={val_loss:.6f}"
        )
        get_run_logger().verbose(
            f"[activation_surrogate] epoch {epoch + 1}/{epochs} "
            f"train={train_loss:.6f} val={val_loss:.6f} "
            f"mask_train={train_mask_frac:.3f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": str(model_type),
            "input_dim": int(sample_q.shape[-1]),
            "output_dim": int(train_ds[0][1].shape[-1]),
            "window_size": int(window_size),
            "normalize_q": bool(normalize_q),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "num_heads": int(num_heads),
            "dim_feedforward": int(dim_feedforward),
            "d_model": int(dim_feedforward) if str(model_type) == "transformer" else 0,
        },
        ckpt_path,
    )

    metrics = {
        "checkpoint": str(ckpt_path),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds) if val_ds is not None else 0,
        "best_val_loss": best_val,
        "history": history,
    }
    (out_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def _apply_json_config(args: argparse.Namespace, config_path: str) -> None:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing config: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key, value in cfg.items():
        if key in {"output", "seed"}:
            continue
        if hasattr(args, key):
            setattr(args, key, value)


def _argv_has_flag(argv: list[str], flag: str) -> bool:
    sep = flag + "="
    return any(a == flag or a.startswith(sep) for a in argv)


def _default_output_dir() -> Path:
    out = os.environ.get("ACTIVATION_SURROGATE_OUTPUT", "").strip()
    if out:
        return Path(out).expanduser()
    return _REPO_ROOT / "results" / f"activation_surrogate_{datetime.now():%Y%m%d_%H%M%S}"


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
    parser.add_argument(
        "--min_window_valid_fraction",
        type=float,
        default=0.35,
        help="Drop training windows with fewer valid Moco label frames.",
    )
    parser.add_argument(
        "--require_activation_mask",
        type=int,
        default=0,
        help="1 = only use B3D files with muscle_activation_mask channel.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    add_run_log_cli_args(parser)
    args = parser.parse_args()

    if not str(args.output).strip():
        out_dir = _default_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir)

    data_root = str(args.data_root).strip()
    if not data_root:
        env_root = os.environ.get("HUMANML3D_ROOT", "").strip()
        if env_root:
            args.data_root = env_root

    cfg_path = str(args.config).strip()
    if not cfg_path:
        default_cfg = _REPO_ROOT / "configs" / "train_surrogate.json"
        if default_cfg.is_file():
            cfg_path = str(default_cfg)
    if cfg_path:
        _apply_json_config(args, cfg_path)

    def _run(_logger: RunLogger) -> None:
        metrics = train_activation_surrogate(
            data_root=args.data_root,
            output=args.output,
            split=args.split,
            val_split=args.val_split,
            window_size=int(args.window_size),
            window_stride=int(args.window_stride),
            normalize_q=bool(int(args.normalize_q)),
            max_motions=int(args.max_motions),
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
            min_window_valid_fraction=float(args.min_window_valid_fraction),
            require_activation_mask=bool(int(args.require_activation_mask)),
            device_name=args.device,
            num_workers=int(args.num_workers),
        )
        _logger.verbose(json.dumps(metrics, indent=2))
        os._exit(0)

    run_logged_main(
        Path(__file__).stem,
        args.log_dir,
        _run,
        argv=sys.argv,
        no_run_log=bool(args.no_run_log),
    )


if __name__ == "__main__":
    main()
