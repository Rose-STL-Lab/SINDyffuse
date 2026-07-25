"""Activation surrogate wrapper for SINDy guidance (muscle actual evaluator)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from surrogate.model import build_activation_surrogate


def _resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        for name in ("latest.pt", "best.pt"):
            candidate = path / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"Missing activation surrogate checkpoint at {path} (expected file or latest.pt in directory)"
    )


class ActivationSurrogateGuidance(nn.Module):
    """Wrap trained surrogate for q → muscle activations inside SINDy guidance."""

    def __init__(
        self,
        model: nn.Module,
        *,
        mean_q: Optional[torch.Tensor] = None,
        std_q: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("mean_q", mean_q if mean_q is not None else torch.zeros(0))
        self.register_buffer("std_q", std_q if std_q is not None else torch.ones(0))

    def _normalize_q(self, q: torch.Tensor) -> torch.Tensor:
        if self.mean_q.numel() == 0:
            return q
        mean = self.mean_q.view(1, 1, -1)
        std = self.std_q.view(1, 1, -1).clamp_min(1e-8)
        return (q - mean) / std

    def predict_activations(self, q: torch.Tensor) -> torch.Tensor:
        """``q`` ``[B, T, D]`` (physical units) → ``[B, T, M]`` in ``[0, 1]``."""
        return self.model(self._normalize_q(q))


def load_activation_surrogate_guidance(
    checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
    data_root: str | None = None,
) -> ActivationSurrogateGuidance:
    """Load ``latest.pt`` (or ``best.pt``) from a surrogate training run."""
    path = _resolve_checkpoint_path(checkpoint)

    try:
        payload: Dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    model = build_activation_surrogate(
        model_type=str(payload.get("model_type", "mlp")),
        input_dim=int(payload["input_dim"]),
        output_dim=int(payload["output_dim"]),
        hidden_dim=int(payload.get("hidden_dim", 256)),
        num_layers=int(payload.get("num_layers", 3)),
        dropout=float(payload.get("dropout", 0.1)),
        num_heads=int(payload.get("num_heads", 4)),
        dim_feedforward=int(payload.get("dim_feedforward", 128)),
        max_seq_len=int(payload.get("window_size", 196)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    mean_q: Optional[torch.Tensor] = None
    std_q: Optional[torch.Tensor] = None
    if bool(payload.get("normalize_q", False)) and data_root:
        import numpy as np

        from common.paths import motion_cache_dir
        from common.skeleton_config import resolve_skeleton

        cache = motion_cache_dir(data_root, skeleton=resolve_skeleton())
        mean_p, std_p = cache / "Mean.npy", cache / "Std.npy"
        if mean_p.is_file() and std_p.is_file():
            mean_q = torch.from_numpy(np.load(mean_p).astype(np.float32))
            std_q = torch.from_numpy(np.load(std_p).astype(np.float32))

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    wrapper = ActivationSurrogateGuidance(
        model.to(dev),
        mean_q=mean_q,
        std_q=std_q,
    )
    return wrapper.to(dev)
