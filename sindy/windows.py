from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
from common.paths import lai_cache_dir
from datasets.lai_cache import SINDY_C_COLS, SINDY_U_COLS, motion_npz_path, read_motion_npz
from datasets.splits import load_split_ids
from nimble.gap_utils import is_nan_placeholder_activations, window_is_valid
from nimble.muscle_b3d import is_zero_placeholder_activations
from sindy.library import ThetaLibrary, ThetaSpec
from sindy.targets import N_SINDY_TARGETS, build_sindy_targets

# Fixed feature names matching nimble.lai_features / sindy.features_from_keypoints
U_FEATURE_NAMES: List[str] = [
    'phase_sin', 'phase_cos', 'cadence_hz',
    'root_speed_x', 'root_speed_y', 'root_speed_z',
    'root_acc_x', 'root_acc_y', 'root_acc_z',
    'pelvis_height',
]
C_FEATURE_NAMES: List[str] = [
    'contact_left', 'contact_right', 'double_support', 'single_support', 'grf_trust_mask',
]


def make_theta_spec(theta_tier: str, include_u: bool, include_c: bool, u_names: List[str]) -> ThetaSpec:
    return ThetaSpec(
        tier=theta_tier,
        include_bias=True,
        include_linear_u=bool(include_u),
        include_linear_c=bool(include_c),
        include_u_times_c=bool(include_u and include_c),
        include_phase_terms=bool(include_u and 'phase_sin' in u_names),
        include_contact_gated=bool(include_c and theta_tier == 'tier3_contact_periodic'),
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
    b3d_path: str  # NPZ path (legacy field name)
    sample_id: str


@dataclass
class SindyIndexStats:
    num_motions_seen: int = 0
    num_motions_kept: int = 0
    num_motions_skipped_missing: int = 0
    num_motions_skipped_short: int = 0
    num_motions_skipped_zero: int = 0
    num_motions_skipped_nan: int = 0
    num_motions_skipped_corrupt: int = 0
    num_windows_skipped_gap: int = 0


class SindyWindowIndex:

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
        skip_invalid_activations: bool = True,
        zero_atol: float = 1e-08,
        min_valid_fraction: float = 0.95,
    ) -> 'SindyWindowIndex':
        root = Path(data_root)
        cache_dir = lai_cache_dir(root)
        if not cache_dir.is_dir():
            raise FileNotFoundError(f'Missing {cache_dir}; run preprocess pipeline first.')
        entries: List[WindowEntry] = []
        stats = SindyIndexStats()
        for sid in load_split_ids(root, split):
            npz_path = motion_npz_path(cache_dir, sid)
            if not npz_path.is_file():
                stats.num_motions_skipped_missing += 1
                continue
            stats.num_motions_seen += 1
            try:
                data = read_motion_npz(npz_path)
            except Exception:
                stats.num_motions_skipped_corrupt += 1
                continue
            act = data['muscle_activations']
            mask = data['muscle_activation_mask']
            tlen = int(data['num_frames'])
            if tlen < int(window_size):
                stats.num_motions_skipped_short += 1
                continue
            if skip_invalid_activations and is_nan_placeholder_activations(act):
                stats.num_motions_skipped_nan += 1
                continue
            if skip_invalid_activations and float(np.sum(mask > 0.5)) <= 0.0:
                stats.num_motions_skipped_nan += 1
                continue
            if skip_zero_placeholders and is_zero_placeholder_activations(act, atol=float(zero_atol)):
                stats.num_motions_skipped_zero += 1
                continue
            stats.num_motions_kept += 1
            for st in _window_starts(tlen, window_size, window_stride):
                if skip_invalid_activations:
                    act_win = act[st : st + window_size]
                    mask_win = mask[st : st + window_size]
                    if not window_is_valid(mask_win, act_win, min_valid_fraction=float(min_valid_fraction)):
                        stats.num_windows_skipped_gap += 1
                        continue
                entries.append(
                    WindowEntry(
                        motion_id=str(sid),
                        start_frame=int(st),
                        b3d_path=str(npz_path),
                        sample_id=f'{sid}:start{st}',
                    )
                )
                if max_samples > 0 and len(entries) >= int(max_samples):
                    break
            if max_samples > 0 and len(entries) >= int(max_samples):
                break
        if not entries:
            raise ValueError(
                f'No SINDy windows indexed from {root} (split={split!r}). '
                f'Re-run preprocess with muscle activations. '
                f'skipped_missing={stats.num_motions_skipped_missing} skipped_zero={stats.num_motions_skipped_zero}'
            )
        print(
            f'[sindy/data] indexed {len(entries)} windows from {stats.num_motions_kept} motions '
            f'(skipped_missing={stats.num_motions_skipped_missing} skipped_zero={stats.num_motions_skipped_zero} '
            f'skipped_nan={stats.num_motions_skipped_nan} skipped_gap_windows={stats.num_windows_skipped_gap} '
            f'skipped_short={stats.num_motions_skipped_short})',
            flush=True,
        )
        return cls(entries, stats=stats)


def compute_window_arrays(
    npz_path: str,
    start_frame: int,
    window_size: int,
    *,
    fps: float,
    compute_bio: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    del fps
    data = read_motion_npz(npz_path)
    st = int(start_frame)
    n = int(window_size)
    u = np.asarray(data['sindy_u'][st : st + n], dtype=np.float32)
    c = np.asarray(data['sindy_c'][st : st + n], dtype=np.float32)
    if u.shape[0] < n or c.shape[0] < n:
        raise RuntimeError(f'Short NPZ window in {npz_path} at frame {start_frame}')
    if u.shape[1] != SINDY_U_COLS or c.shape[1] != SINDY_C_COLS:
        raise ValueError(f'Unexpected sindy feature dims u={u.shape} c={c.shape}')
    un, cn = list(U_FEATURE_NAMES), list(C_FEATURE_NAMES)
    if compute_bio:
        bio = np.asarray(data['guidance_bio'][st : st + n], dtype=np.float32)
        act = np.asarray(data['muscle_activations'][st : st + n], dtype=np.float32)
        y = build_sindy_targets(bio, act)
    else:
        y = np.zeros((max(0, n - 1), N_SINDY_TARGETS), dtype=np.float32)
    if y.shape[1] != N_SINDY_TARGETS:
        raise ValueError(f'Expected y with {N_SINDY_TARGETS} targets, got {y.shape}')
    return (u, c, y, un, cn)


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
    theta_scaler = StandardScaler()
    y_scaler = StandardScaler()
    theta_lib = ThetaLibrary(spec=theta_spec)
    u_names: List[str] = []
    c_names: List[str] = []
    feature_names: List[str] = []
    target_dim = 0
    length = int(window_size) - 1
    for i, entry in enumerate(index.entries):
        u, c, y, un, cn = compute_window_arrays(entry.b3d_path, entry.start_frame, window_size, fps=fps)
        if not u_names:
            u_names, c_names = (un, cn)
        target_dim = int(y.shape[1])
        u_in = u[:-1, :] if include_u else None
        c_in = c[:-1, :] if include_c else None
        theta_flat, feature_names = theta_lib.build(
            u=u_in, c=c_in, u_names=u_names if include_u else [], c_names=c_names if include_c else []
        )
        if int(theta_flat.shape[0]) != length:
            raise ValueError(f'theta length {theta_flat.shape[0]} != {length}')
        theta_scaler.partial_fit(theta_flat)
        y_scaler.partial_fit(y.reshape(-1, target_dim))
        if log_every > 0 and (i + 1) % int(log_every) == 0:
            print(f'[sindy/data] scaler pass {i + 1}/{len(index.entries)}', flush=True)
    return (index, theta_scaler, y_scaler, u_names, c_names, feature_names, target_dim)


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
    zero_atol: float = 1e-08,
):
    index = SindyWindowIndex.build(
        data_root,
        split,
        window_size=window_size,
        window_stride=window_stride,
        max_samples=max_samples,
        skip_zero_placeholders=skip_zero_placeholders,
        zero_atol=zero_atol,
    )
    u_list, c_list, y_list, sid_list = ([], [], [], [])
    u_names: List[str] = []
    c_names: List[str] = []
    for i, entry in enumerate(index.entries):
        u, c, y, un, cn = compute_window_arrays(
            entry.b3d_path, entry.start_frame, window_size, fps=fps, compute_bio=compute_bio
        )
        if not u_names:
            u_names, c_names = (un, cn)
        u_list.append(u)
        c_list.append(c)
        y_list.append(y)
        sid_list.append(entry.sample_id)
        n = len(sid_list)
        if log_every > 0 and n % int(log_every) == 0:
            print(f'[sindy/data] preloaded {n}/{len(index.entries)} windows', flush=True)
    if not u_list:
        raise ValueError(f'No NPZ windows collected from {data_root}')
    u = np.stack(u_list, axis=0).astype(np.float32)
    c = np.stack(c_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.float32)
    return (u, c, y, u_names, c_names, np.array(sid_list, dtype=object))
