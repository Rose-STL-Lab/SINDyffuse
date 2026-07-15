"""Train text→Xi SINDy model on Nimble L_bio targets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import clip  # type: ignore
import joblib
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from common.distributed import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_process,
    log_main,
    model_state_dict,
    resolve_train_device,
    seed_all,
    setup_spawn_if_distributed,
    shard_indices,
    unwrap_module as unwrap_train_module,
    wrap_ddp,
)
from common.paths import default_humanml3d_root, humanml3d_text_dir, nimble_b3d_dir
from common.run_logging import RunLogger, add_run_log_cli_args, get_run_logger, run_logged_main
from nimble.channels import BIOMECH_COMPONENT_KEYS

from sindy.dataset import SindyWindowDataset, prepare_lazy_sindy_data
from sindy.windows import collect_windows, make_theta_spec
from sindy.library import ThetaLibrary
from sindy.model import TextToXi, predict_from_xi, xi_temporal_smoothness


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
    model, _ = clip.load(model_name, device=device, jit=False)
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
    rng: np.random.Generator,
    fetch_batch: Any,
    run_validation: bool = True,
) -> Tuple[float, Optional[Dict[str, torch.Tensor]]]:
    """Shared epoch loop for preloaded tensors or lazy dataset batches."""
    best_val, best_state = float("inf"), None
    for ep in range(1, int(epochs) + 1):
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
                loss_fit = loss_fit + torch.mean((pred - y_b[bi : bi + 1]) ** 2)
                xi_tensor = xi_out[0] if isinstance(xi_out, tuple) else xi_out
                loss_sparse = loss_sparse + torch.mean(torch.abs(xi_tensor))
                loss_smooth = loss_smooth + xi_temporal_smoothness(xi_out)
            denom = float(max(1, len(bidx)))
            loss = loss_fit / denom + float(lambda_l1) * (loss_sparse / denom) + float(lambda_xi_smooth) * (loss_smooth / denom)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_loss = float("inf")
        if run_validation and len(val_idx) > 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for sample_idx in val_idx:
                    cap = captions[int(sample_idx)]
                    e = emb_t[cap_to_idx[cap] : cap_to_idx[cap] + 1]
                    theta_b, y_b = fetch_batch([int(sample_idx)])
                    xi_out = model(e, seq_len=int(length))
                    pred = predict_from_xi(theta_b, xi_out)
                    val_loss += float(torch.mean((pred - y_b) ** 2).cpu())
            val_loss /= max(1, len(val_idx))
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model_state_dict(model).items()}
        if (ep % max(1, epochs // 20) == 0 or ep == 1) and is_main_process():
            vl = val_loss if run_validation else float("nan")
            from common.run_logging import get_run_logger

            get_run_logger().progress(
                f"epoch={ep}/{epochs} train_loss={np.mean(losses):.6f} val_loss={vl:.6f}"
            )
    return best_val, best_state


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
    epochs: int,
    batch_size: int,
    lambda_l1: float,
    lambda_xi_smooth: float,
    hidden_dim: int,
    num_experts: int,
    fallback_caption_weight: float,
    clip_model_name: str,
    clip_text_batch_size: int,
    device_name: str,
    bio_log_every: int,
    preload: bool = False,
    distributed_cfg: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    use_ddp = init_distributed(distributed_cfg=distributed_cfg)
    seed_all(int(seed))
    root = Path(data_root)
    if not nimble_b3d_dir(root).is_dir():
        raise FileNotFoundError(
            f"Nimble B3D cache required at {nimble_b3d_dir(root)}. "
            f"Run preprocess_nimble.py first."
        )
    bio_keys = list(BIOMECH_COMPONENT_KEYS)
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
        )
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
    ).to(device)
    model = wrap_ddp(model, find_unused_parameters=find_unused)
    opt = optim.AdamW(model.parameters(), lr=float(lr))
    log_main(
        f"[sindy/train] device={device} world_size={get_world_size()} "
        f"distributed={use_ddp} train_windows={len(tr_idx)} per_gpu_batch={batch_size}"
    )
    emb_t = torch.tensor(unique_emb, dtype=torch.float32, device=device)

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

    best_val, best_state = _run_training_loop(
        model=model,
        opt=opt,
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
        rng=rng,
        fetch_batch=fetch_batch,
        run_validation=is_main_process(),
    )

    if best_state is not None:
        unwrap_train_module(model).load_state_dict(best_state)

    out = Path(output)
    if is_main_process():
        out.mkdir(parents=True, exist_ok=True)
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
            "bio_channel_names": bio_keys,
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
            "window_size": int(window_size),
            "window_stride": int(window_stride),
            "theta_tier": theta_tier,
            "preload": bool(preload),
        },
        "metrics": {"best_val_loss": float(best_val)},
    }
    with open(out / "sild_text_train_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
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
    setup_spawn_if_distributed()
    parser = argparse.ArgumentParser(description="Train SINDy text→Xi on Nimble L_bio targets")
    parser.add_argument(
        "--config",
        default="",
        help="Path to train_sindy.json (default: configs/train_sindy.json next to repo root)",
    )
    parser.add_argument("--data_root", default=default_humanml3d_root())
    parser.add_argument("--split", default="train")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--window_stride", type=int, default=16)
    parser.add_argument("--theta_tier", default="tier2_moderate")
    parser.add_argument("--include_u", type=int, default=1)
    parser.add_argument("--include_c", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lambda_l1", type=float, default=1e-4)
    parser.add_argument("--lambda_xi_smooth", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_experts", type=int, default=1)
    parser.add_argument("--fallback_caption_weight", type=float, default=0.2)
    parser.add_argument("--clip_model_name", default="ViT-B/32")
    parser.add_argument("--clip_text_batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bio_log_every", type=int, default=50)
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load all B3D windows into RAM before training (default: read on demand)",
    )
    add_run_log_cli_args(parser)
    args = parser.parse_args()
    cfg_path = str(args.config).strip()
    dist_cfg: Dict[str, Any] = {}
    full_cfg: Dict[str, Any] = {}
    if not cfg_path:
        repo_root = Path(__file__).resolve().parents[1]
        default_cfg = repo_root / "configs" / "train_sindy.json"
        if default_cfg.is_file():
            cfg_path = str(default_cfg)
    if cfg_path:
        _apply_json_config(args, cfg_path)
        full_cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        if isinstance(full_cfg.get("distributed"), dict):
            dist_cfg = full_cfg["distributed"]
    else:
        full_cfg = {}
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
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lambda_l1=float(args.lambda_l1),
                lambda_xi_smooth=float(args.lambda_xi_smooth),
                hidden_dim=int(args.hidden_dim),
                num_experts=int(args.num_experts),
                fallback_caption_weight=float(args.fallback_caption_weight),
                clip_model_name=args.clip_model_name,
                clip_text_batch_size=int(args.clip_text_batch_size),
                device_name=args.device,
                bio_log_every=int(args.bio_log_every),
                preload=bool(args.preload),
                distributed_cfg=dist_cfg,
                seed=int(full_cfg.get("seed", 42)) if cfg_path else 42,
            )
            if is_main_process():
                _logger.verbose(json.dumps(metrics, indent=2))
        finally:
            cleanup_distributed()
        try:
            from nimble.physics import clear_cache

            clear_cache()
        except Exception:
            pass
        os._exit(0)

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
