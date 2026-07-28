"""On-demand SINDy window dataset (used when ``preload=False``)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from sindy.library import ThetaLibrary, ThetaSpec
from sindy.windows import SindyWindowIndex, compute_window_arrays, fit_window_scalers, make_theta_spec


class SindyWindowDataset(Dataset):
    """Load one SINDy window per ``__getitem__`` (B3D read + bio features on demand)."""

    def __init__(
        self,
        data_root: str,
        index: SindyWindowIndex,
        *,
        window_size: int,
        fps: float,
        theta_spec: ThetaSpec,
        include_u: bool,
        include_c: bool,
        theta_scaler: StandardScaler,
        y_scaler: StandardScaler,
        u_names: List[str],
        c_names: List[str],
        feature_names: List[str],
        target_dim: int,
    ):
        self.data_root = Path(data_root)
        self.index = index
        self.window_size = int(window_size)
        self.fps = float(fps)
        self.theta_spec = theta_spec
        self.include_u = bool(include_u)
        self.include_c = bool(include_c)
        self.theta_scaler = theta_scaler
        self.y_scaler = y_scaler
        self.u_names = list(u_names)
        self.c_names = list(c_names)
        self.feature_names = list(feature_names)
        self.target_dim = int(target_dim)
        self.theta_lib = ThetaLibrary(spec=theta_spec)
        self._length = int(window_size) - 1

    def __len__(self) -> int:
        return len(self.index.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.index.entries[int(idx)]
        u, c, y, _, _ = compute_window_arrays(
            entry.cache_path,
            entry.start_frame,
            self.window_size,
            fps=self.fps,
        )
        u_in = u[:-1, :] if self.include_u else None
        c_in = c[:-1, :] if self.include_c else None
        theta_flat, _ = self.theta_lib.build(
            u=u_in,
            c=c_in,
            u_names=self.u_names if self.include_u else [],
            c_names=self.c_names if self.include_c else [],
        )
        f = int(theta_flat.shape[1])
        theta = theta_flat.reshape(self._length, f).astype(np.float32)
        theta_s = self.theta_scaler.transform(theta).astype(np.float32)
        y_s = self.y_scaler.transform(y.astype(np.float32)).astype(np.float32)
        return {
            "theta": torch.tensor(theta_s, dtype=torch.float32),
            "y": torch.tensor(y_s, dtype=torch.float32),
            "sample_id": entry.sample_id,
        }


def prepare_lazy_sindy_data(
    data_root: str,
    split: str,
    *,
    fps: float,
    window_size: int,
    window_stride: int,
    max_samples: int,
    theta_tier: str,
    include_u: bool,
    include_c: bool,
    log_every: int = 50,
    skip_zero_placeholders: bool = True,
    zero_atol: float = 1e-8,
) -> Tuple[SindyWindowIndex, StandardScaler, StandardScaler, List[str], List[str], List[str], int, ThetaSpec]:
    """Build window index and fit scalers with a single streaming pass."""
    index = SindyWindowIndex.build(
        data_root,
        split,
        window_size=window_size,
        window_stride=window_stride,
        max_samples=max_samples,
        skip_zero_placeholders=skip_zero_placeholders,
        zero_atol=zero_atol,
    )
    print(f"[sindy/data] lazy mode: {len(index.entries)} windows indexed", flush=True)
    first = index.entries[0]
    _u, _c, _y, u_names, c_names = compute_window_arrays(
        first.cache_path,
        first.start_frame,
        window_size,
        fps=fps,
    )[:4]
    spec = make_theta_spec(theta_tier, include_u, include_c, u_names)
    index, theta_scaler, y_scaler, u_names, c_names, feature_names, target_dim = fit_window_scalers(
        index,
        window_size=window_size,
        fps=fps,
        theta_spec=spec,
        include_u=include_u,
        include_c=include_c,
        log_every=log_every,
    )
    return index, theta_scaler, y_scaler, u_names, c_names, feature_names, target_dim, spec
