from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def resolve_torch_device(device: Optional[str] = None) -> torch.device:
    name = (device or "auto").strip().lower()
    if name in {"auto", "", "none"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {device!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
