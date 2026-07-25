"""Collect training windows from Nimble B3D cache for SINDy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import nimblephysics as nimble
from sklearn.preprocessing import StandardScaler

from common.paths import mint_cache_dir, nimble_b3d_dir
from common.skeleton_config import SkeletonKind, resolve_skeleton
from datasets.nimble_dataset import read_q_frames
from datasets.splits import kinematics_pass_index, load_split_ids
from nimble.b3d_io import (
    b3d_has_muscle_activations,
    b3d_has_sindyffuse_custom_values,
    read_guidance_features_frames,
    read_muscle_activations_frames,
    read_sindy_features_frames,
    warn_missing_custom_once,
)
from nimble.muscle_b3d import is_zero_placeholder_activations
from sindy.library import ThetaLibrary, ThetaSpec
from sindy.targets import (
    N_SINDY_TARGETS,
    build_sindy_targets,
)


def make_theta_spec(theta_tier: str, include_u: bool, include_c: bool, u_names: List[str]) -> ThetaSpec:
    return ThetaSpec(
        tier=theta_tier,
        include_bias=True,
        include_linear_u=bool(include_u),
        include_linear_c=bool(include_c),
        include_u_times_c=bool(include_u and include_c),
        include_phase_terms=bool(include_u and ("phase_sin" in u_names)),
        include_contact_gated=bool(include_c and theta_tier == "tier3_contact_periodic"),
        max_cross_terms=30,
    )


def _window_starts(length: int, window_size: int, window_stride: int) -> List[int]:
    if length < window_size:
        return []
    return list(range(0, length - window_size + 1, window_stride))


@dataclass(frozen=True)
class WindowEntry:
    motion_id: str
    start_frame: int
    b3d_path: str
    sample_id: str


@dataclass
class SindyIndexStats:
    num_motions_seen: int = 0
    num_motions_kept: int = 0
    num_motions_skipped_missing: int = 0
    num_motions_skipped_short: int = 0
    num_motions_skipped_zero: int = 0


class SindyWindowIndex:
    """List of strided B3D windows for a split (metadata only)."""

    def __init__(self, entries: List[WindowEntry], *, stats: SindyIndexStats | None = None):
        self.entries = list(entries)
        self.stats = stats or SindyIndexStats()

    @classmethod
    def build(
        cls,
        data_root: str,
        split: str,
        *,
        window_size: int,
        window_stride: int,
        max_samples: int,
        skip_zero_placeholders: bool = True,
        zero_atol: float = 1e-8,
    ) -> "SindyWindowIndex":
        root = Path(data_root)
        b3d_dir = nimble_b3d_dir(root)
        if not b3d_dir.is_dir():
            raise FileNotFoundError(f"Missing {b3d_dir}; run preprocess_nimble.py first.")

        entries: List[WindowEntry] = []
        stats = SindyIndexStats()
        for sid in load_split_ids(root, split):
            b3d_path = b3d_dir / f"{sid}.b3d"
            if not b3d_path.is_file():
                stats.num_motions_skipped_missing += 1
                continue
            stats.num_motions_seen += 1
            subj = nimble.biomechanics.SubjectOnDisk(str(b3d_path))
            if not b3d_has_sindyffuse_custom_values(subj):
                stats.num_motions_skipped_missing += 1
                continue
            if not b3d_has_muscle_activations(subj):
                stats.num_motions_skipped_missing += 1
                continue

            trial = 0
            tlen = int(subj.getTrialLength(trial))
            if tlen < int(window_size):
                stats.num_motions_skipped_short += 1
                continue

            if skip_zero_placeholders:
                act = read_muscle_activations_frames(subj, trial, 0, tlen)
                if is_zero_placeholder_activations(act, atol=float(zero_atol)):
                    stats.num_motions_skipped_zero += 1
                    continue

            stats.num_motions_kept += 1
            for st in _window_starts(tlen, window_size, window_stride):
                entries.append(
                    WindowEntry(
                        motion_id=str(sid),
                        start_frame=int(st),
                        b3d_path=str(b3d_path),
                        sample_id=f"{sid}:start{st}",
                    )
                )
                if max_samples > 0 and len(entries) >= int(max_samples):
                    break
            if max_samples > 0 and len(entries) >= int(max_samples):
                break

        if not entries:
            raise ValueError(
                f"No SINDy windows indexed from {root} (split={split!r}). "
                f"Re-run preprocess_nimble.py with muscle activations. "
                f"skipped_missing={stats.num_motions_skipped_missing} "
                f"skipped_zero={stats.num_motions_skipped_zero}"
            )
        print(
            f"[sindy/data] indexed {len(entries)} windows from {stats.num_motions_kept} motions "
            f"(skipped_missing={stats.num_motions_skipped_missing} "
            f"skipped_zero={stats.num_motions_skipped_zero} "
            f"skipped_short={stats.num_motions_skipped_short})",
            flush=True,
        )
        return cls(entries, stats=stats)

    @classmethod
    def build_mint(
        cls,
        data_root: str,
        split: str,
        *,
        window_size: int,
        window_stride: int,
        max_samples: int,
        skip_zero_placeholders: bool = True,
        zero_atol: float = 1e-8,
    ) -> "SindyWindowIndex":
        from mint.cache_schema import cache_has_labels, read_motion_cache
        from mint.muscle_schema import validate_activation_matrix

        root = Path(data_root)
        cache_dir = mint_cache_dir(root)
        if not cache_dir.is_dir():
            raise FileNotFoundError(f"Missing {cache_dir}; run preprocess_mint.py first.")

        entries: List[WindowEntry] = []
        stats = SindyIndexStats()
        for sid in load_split_ids(root, split):
            npz_path = cache_dir / f"{sid}.npz"
            if not npz_path.is_file():
                stats.num_motions_skipped_missing += 1
                continue
            stats.num_motions_seen += 1
            data = read_motion_cache(npz_path)
            tlen = int(data["q"].shape[0])
            if tlen < int(window_size):
                stats.num_motions_skipped_short += 1
                continue
            if not cache_has_labels(npz_path):
                stats.num_motions_skipped_missing += 1
                continue
            if skip_zero_placeholders:
                act = data["muscle_activations"]
                if validate_activation_matrix(act, atol=float(zero_atol)):
                    stats.num_motions_skipped_zero += 1
                    continue

            stats.num_motions_kept += 1
            for st in _window_starts(tlen, window_size, window_stride):
                entries.append(
                    WindowEntry(
                        motion_id=str(sid),
                        start_frame=int(st),
                        b3d_path=str(npz_path),
                        sample_id=f"{sid}:start{st}",
                    )
                )
                if max_samples > 0 and len(entries) >= int(max_samples):
                    break
            if max_samples > 0 and len(entries) >= int(max_samples):
                break

        if not entries:
            raise ValueError(
                f"No MinT SINDy windows indexed from {root} (split={split!r}). "
                f"Run preprocess_mint.py with MinT labels."
            )
        print(
            f"[sindy/data] indexed {len(entries)} MinT windows from {stats.num_motions_kept} motions",
            flush=True,
        )
        return cls(entries, stats=stats)


def compute_mint_window_arrays(
    npz_path: str,
    start_frame: int,
    window_size: int,
    *,
    fps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    from mint.cache_schema import read_motion_cache
    from mint.features import features_from_mint_q

    data = read_motion_cache(npz_path)
    st = int(start_frame)
    n = int(window_size)
    q = data["q"][st : st + n].astype(np.float32)
    bio = data["guidance_features"][st : st + n].astype(np.float32)
    act = data["muscle_activations"][st : st + n].astype(np.float32)
    if "sindy_features" in data:
        sc = data["sindy_features"][st : st + n].astype(np.float32)
        u_dim = 10
        u = sc[:, :u_dim]
        c = sc[:, u_dim:]
        from mint.features import features_from_mint_q as _ff

        _, _, un, cn = _ff(q, fps=fps)
    else:
        u, c, un, cn = features_from_mint_q(q, fps=fps)
    y = build_sindy_targets(bio, act)
    if y.shape[1] != N_SINDY_TARGETS:
        raise ValueError(f"Expected y with {N_SINDY_TARGETS} targets, got {y.shape}")
    return u, c, y, un, cn


def compute_window_arrays(
    cache_path: str,
    start_frame: int,
    window_size: int,
    *,
    fps: float,
    compute_bio: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """Read one window from B3D or MinT NPZ and compute SINDy ``u``, ``c``, ``y``."""
    if str(cache_path).endswith(".npz"):
        return compute_mint_window_arrays(cache_path, start_frame, window_size, fps=fps)

    b3d_path = cache_path
    subj = nimble.biomechanics.SubjectOnDisk(str(b3d_path))
    trial = 0
    kin = kinematics_pass_index(subj, trial)
    win_q = read_q_frames(subj, trial, int(start_frame), int(window_size), kin=kin)
    if win_q.shape[0] < int(window_size):
        raise RuntimeError(f"Short B3D window in {b3d_path} at frame {start_frame}")
    if b3d_has_sindyffuse_custom_values(subj):
        u, c = read_sindy_features_frames(subj, trial, int(start_frame), int(window_size))
        from nimble.b3d_schema import C_FEATURE_NAMES, U_FEATURE_NAMES

        un, cn = list(U_FEATURE_NAMES), list(C_FEATURE_NAMES)
        if compute_bio:
            bio = read_guidance_features_frames(subj, trial, int(start_frame), int(window_size))
            act = read_muscle_activations_frames(subj, trial, int(start_frame), int(window_size))
            y = build_sindy_targets(bio, act)
        else:
            y = np.zeros((max(0, window_size - 1), N_SINDY_TARGETS), dtype=np.float32)
    else:
        warn_missing_custom_once(str(b3d_path), "guidance_features/sindy_features/muscle_activations")
        raise RuntimeError(
            f"B3D {b3d_path} missing SINDyffuse custom values with muscle activations; "
            f"re-run preprocess_nimble.py with --activation_method moco_track or static_optimization."
        )
    if y.shape[1] != N_SINDY_TARGETS:
        raise ValueError(f"Expected y with {N_SINDY_TARGETS} targets, got {y.shape}")
    return u, c, y, un, cn


def fit_window_scalers(
    index: SindyWindowIndex,
    *,
    window_size: int,
    fps: float,
    theta_spec: ThetaSpec,
    include_u: bool,
    include_c: bool,
    log_every: int = 50,
) -> Tuple[SindyWindowIndex, StandardScaler, StandardScaler, List[str], List[str], List[str], int]:
    """Stream windows once to fit ``theta`` and ``y`` scalers."""
    theta_scaler = StandardScaler()
    y_scaler = StandardScaler()
    theta_lib = ThetaLibrary(spec=theta_spec)
    u_names: List[str] = []
    c_names: List[str] = []
    feature_names: List[str] = []
    target_dim = 0
    length = int(window_size) - 1

    for i, entry in enumerate(index.entries):
        u, c, y, un, cn = compute_window_arrays(
            entry.b3d_path,
            entry.start_frame,
            window_size,
            fps=fps,
        )
        if not u_names:
            u_names, c_names = un, cn
        target_dim = int(y.shape[1])
        u_in = u[:-1, :] if include_u else None
        c_in = c[:-1, :] if include_c else None
        theta_flat, feature_names = theta_lib.build(
            u=u_in,
            c=c_in,
            u_names=u_names if include_u else [],
            c_names=c_names if include_c else [],
        )
        if int(theta_flat.shape[0]) != length:
            raise ValueError(f"theta length {theta_flat.shape[0]} != {length}")
        theta_scaler.partial_fit(theta_flat)
        y_scaler.partial_fit(y.reshape(-1, target_dim))
        if log_every > 0 and (i + 1) % int(log_every) == 0:
            print(f"[sindy/data] scaler pass {i + 1}/{len(index.entries)}", flush=True)

    return index, theta_scaler, y_scaler, u_names, c_names, feature_names, target_dim


def collect_windows(
    data_root: str,
    split: str,
    fps: float,
    window_size: int,
    window_stride: int,
    max_samples: int,
    *,
    compute_bio: bool = True,
    log_every: int = 50,
    skip_zero_placeholders: bool = True,
    zero_atol: float = 1e-8,
    skeleton: str | None = None,
):
    """Preload all SINDy windows into memory (``preload=True`` path)."""
    sk = resolve_skeleton(skeleton)
    if sk == SkeletonKind.MINT:
        index = SindyWindowIndex.build_mint(
            data_root,
            split,
            window_size=window_size,
            window_stride=window_stride,
            max_samples=max_samples,
            skip_zero_placeholders=skip_zero_placeholders,
            zero_atol=zero_atol,
        )
        u_list, c_list, y_list, sid_list = [], [], [], []
        u_names: List[str] = []
        c_names: List[str] = []
        for i, entry in enumerate(index.entries):
            u, c, y, un, cn = compute_mint_window_arrays(
                entry.b3d_path,
                entry.start_frame,
                window_size,
                fps=fps,
            )
            if not u_names:
                u_names, c_names = un, cn
            u_list.append(u)
            c_list.append(c)
            y_list.append(y)
            sid_list.append(entry.sample_id)
            if log_every > 0 and len(sid_list) % int(log_every) == 0:
                print(f"[sindy/data] preloaded {len(sid_list)}/{len(index.entries)} MinT windows", flush=True)
        u = np.stack(u_list, axis=0).astype(np.float32)
        c = np.stack(c_list, axis=0).astype(np.float32)
        y = np.stack(y_list, axis=0).astype(np.float32)
        return u, c, y, u_names, c_names, np.array(sid_list, dtype=object)

    index = SindyWindowIndex.build(
        data_root,
        split,
        window_size=window_size,
        window_stride=window_stride,
        max_samples=max_samples,
        skip_zero_placeholders=skip_zero_placeholders,
        zero_atol=zero_atol,
    )
    u_list, c_list, y_list, sid_list = [], [], [], []
    u_names: List[str] = []
    c_names: List[str] = []

    for i, entry in enumerate(index.entries):
        u, c, y, un, cn = compute_window_arrays(
            entry.b3d_path,
            entry.start_frame,
            window_size,
            fps=fps,
            compute_bio=compute_bio,
        )
        if not u_names:
            u_names, c_names = un, cn
        u_list.append(u)
        c_list.append(c)
        y_list.append(y)
        sid_list.append(entry.sample_id)
        n = len(sid_list)
        if log_every > 0 and n % int(log_every) == 0:
            print(f"[sindy/data] preloaded {n}/{len(index.entries)} windows", flush=True)

    if not u_list:
        raise ValueError(f"No B3D windows collected from {data_root}")

    u = np.stack(u_list, axis=0).astype(np.float32)
    c = np.stack(c_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.float32)
    try:
        from nimble.physics import clear_cache

        clear_cache()
    except Exception:
        pass
    return u, c, y, u_names, c_names, np.array(sid_list, dtype=object)
