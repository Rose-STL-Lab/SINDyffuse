"""MinT-style segmented MocoTrack: plan windows, ground offset, stitch outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from nimble.muscle_activation import (
    MuscleActivationConfig,
    MuscleActivationResult,
    muscle_names,
    opensim_quiet,
    rajagopal_model_path,
)
from nimble.rajagopal_coord_map import RAJAGOPAL_NIMBLE_DOF_NAMES
from nimble.rajagopal_kin import foot_body_indices

PELVIS_TY_COL = int(RAJAGOPAL_NIMBLE_DOF_NAMES.index("ground_pelvis_4"))
SIM_GRF_COLS = 18

# MinT-like column order (force/torque per foot, world frame).
SIM_GRF_CHANNEL_NAMES: Tuple[str, ...] = (
    "ground_force_left_vx",
    "ground_force_left_vy",
    "ground_force_left_vz",
    "ground_torque_left_x",
    "ground_torque_left_y",
    "ground_torque_left_z",
    "ground_force_right_vx",
    "ground_force_right_vy",
    "ground_force_right_vz",
    "ground_torque_right_x",
    "ground_torque_right_y",
    "ground_torque_right_z",
    "ground_force_vertical",
    "ground_force_left_norm",
    "ground_force_right_norm",
    "ground_torque_left_norm",
    "ground_torque_right_norm",
    "grf_trust",
)


@dataclass(frozen=True)
class SegmentSpec:
    """One MinT-style segment: solve window includes buffers; core is kept output."""

    index: int
    solve_start: int
    solve_end: int
    core_start: int
    core_end: int

    @property
    def solve_len(self) -> int:
        return int(self.solve_end - self.solve_start)

    @property
    def core_len(self) -> int:
        return int(self.core_end - self.core_start)


def segment_frame_counts(
    fps: float,
    *,
    core_s: float = 1.4,
    buffer_s: float = 0.14,
) -> Tuple[int, int]:
    """Return ``(core_frames, buffer_frames)`` from durations."""
    rate = max(float(fps), 1e-8)
    core = max(1, int(round(float(core_s) * rate)))
    buf = max(0, int(round(float(buffer_s) * rate)))
    return core, buf


def plan_moco_segments(
    t_len: int,
    fps: float,
    *,
    core_s: float = 1.4,
    buffer_s: float = 0.14,
) -> List[SegmentSpec]:
    """Tile non-overlapping core segments with buffered solve windows."""
    t_len = int(t_len)
    if t_len < 2:
        return []
    core_frames, buffer_frames = segment_frame_counts(
        fps, core_s=core_s, buffer_s=buffer_s
    )
    solve_min = core_frames + 2 * buffer_frames
    if t_len < solve_min:
        return [
            SegmentSpec(
                index=0,
                solve_start=0,
                solve_end=t_len,
                core_start=0,
                core_end=t_len,
            )
        ]

    specs: List[SegmentSpec] = []
    core_start = 0
    seg_idx = 0
    while core_start < t_len:
        core_end = min(t_len, core_start + core_frames)
        if core_end <= core_start:
            break
        solve_start = max(0, core_start - buffer_frames)
        solve_end = min(t_len, core_end + buffer_frames)
        specs.append(
            SegmentSpec(
                index=seg_idx,
                solve_start=solve_start,
                solve_end=solve_end,
                core_start=core_start,
                core_end=core_end,
            )
        )
        seg_idx += 1
        core_start = core_end
    return specs


def _body_origin_y(sk: Any, body_idx: int, q_row: np.ndarray) -> float:
    sk.setPositions(np.asarray(q_row, dtype=np.float64).reshape(-1))
    sk.computeForwardKinematics()
    tr = sk.getBodyNode(int(body_idx)).getWorldTransform()
    if hasattr(tr, "translation"):
        pos = np.asarray(tr.translation(), dtype=np.float64).reshape(3)
    else:
        pos = np.asarray(tr, dtype=np.float64).reshape(3)
    return float(pos[1])


def foot_contact_point_min_y(
    sk: Any,
    q: np.ndarray,
    *,
    foot_body_names: Sequence[str] = ("calcn_l", "calcn_r"),
    sphere_offset_y_m: float = -0.02,
) -> float:
    """Minimum foot contact-sphere center height over the clip."""
    left_idx, right_idx = foot_body_indices(sk)
    body_idxs = (left_idx, right_idx)
    q_arr = np.asarray(q, dtype=np.float64)
    min_y = float("inf")
    for t in range(int(q_arr.shape[0])):
        for bi in body_idxs:
            y = _body_origin_y(sk, bi, q_arr[t]) + float(sphere_offset_y_m)
            min_y = min(min_y, y)
    return float(min_y) if math.isfinite(min_y) else 0.0


def apply_ground_offset_q(
    q: np.ndarray,
    sk: Any,
    cfg: MuscleActivationConfig,
    *,
    foot_body_names: Sequence[str] = ("calcn_l", "calcn_r"),
) -> Tuple[np.ndarray, float]:
    """Shift ``pelvis_ty`` so foot contact spheres sit near ground (MinT-style)."""
    q_arr = np.asarray(q, dtype=np.float64).copy()
    if q_arr.ndim != 2 or q_arr.shape[1] <= PELVIS_TY_COL:
        return q_arr, 0.0

    offset_y = float(cfg.moco_contact_sphere_offset_y_m)
    radius = float(cfg.moco_contact_sphere_radius_m)
    target_min_y = float(radius) + max(0.0, -offset_y) + 0.005
    min_y = foot_contact_point_min_y(
        sk,
        q_arr,
        foot_body_names=foot_body_names,
        sphere_offset_y_m=offset_y,
    )
    shift = float(target_min_y - min_y)
    if abs(shift) > 1e-9:
        q_arr[:, PELVIS_TY_COL] += shift
    return q_arr, shift


def stitch_segment_values(
    t_len: int,
    segments: Sequence[SegmentSpec],
    core_values: Sequence[np.ndarray],
    *,
    blend_frames: int,
    stitch_seams: bool,
) -> np.ndarray:
    """Stitch per-segment core arrays ``[core_len, C]`` into ``[T, C]``."""
    if not segments:
        raise ValueError("stitch_segment_values requires at least one segment")
    t_len = int(t_len)
    sample = np.asarray(core_values[0], dtype=np.float64)
    if sample.ndim != 2:
        raise ValueError(f"Expected core values [core, C], got {sample.shape}")
    n_cols = int(sample.shape[1])
    out = np.full((t_len, n_cols), np.nan, dtype=np.float64)
    weight = np.zeros((t_len,), dtype=np.float64)

    blend = max(0, int(blend_frames))
    for spec, core in zip(segments, core_values):
        arr = np.asarray(core, dtype=np.float64)
        if arr.shape != (spec.core_len, n_cols):
            raise ValueError(
                f"Segment {spec.index} shape {arr.shape} != ({spec.core_len}, {n_cols})"
            )
        for local_t, global_t in enumerate(range(spec.core_start, spec.core_end)):
            w = 1.0
            if stitch_seams and blend > 0 and spec.index > 0 and local_t < blend:
                alpha = float(local_t + 1) / float(blend + 1)
                w = alpha
            elif stitch_seams and blend > 0 and spec.index < len(segments) - 1:
                dist_end = spec.core_len - 1 - local_t
                if dist_end < blend:
                    alpha = float(dist_end + 1) / float(blend + 1)
                    w = min(w, alpha)
            prev_w = weight[global_t]
            if prev_w <= 0.0 or not np.isfinite(prev_w):
                out[global_t] = arr[local_t]
                weight[global_t] = w
            else:
                total = prev_w + w
                if total > 0.0:
                    out[global_t] = (out[global_t] * prev_w + arr[local_t] * w) / total
                    weight[global_t] = total
                else:
                    out[global_t] = arr[local_t]
                    weight[global_t] = w
    return out.astype(np.float32)


def stitch_segment_mask(
    t_len: int,
    segments: Sequence[SegmentSpec],
    segment_ok: Sequence[bool],
) -> np.ndarray:
    """Binary validity mask ``[T]`` from per-segment solve success."""
    mask = np.zeros((int(t_len),), dtype=np.float32)
    for spec, ok in zip(segments, segment_ok):
        if not ok:
            continue
        mask[spec.core_start : spec.core_end] = 1.0
    return mask


def run_moco_track_segmented(
    q: np.ndarray,
    *,
    cfg: MuscleActivationConfig,
    work_dir: Path,
    skeleton: Any | None = None,
) -> MuscleActivationResult:
    """Run MocoTrack on MinT-style segments and stitch activations + GRF."""
    from nimble.moco_track import (
        _effective_mesh_interval,
        _solve_moco_track,
        _track_config,
        build_rajagopal_coord_mapping,
        prepare_rajagopal_moco_track_model,
    )

    arr = np.asarray(q, dtype=np.float64)
    t_len = int(arr.shape[0])
    segments = plan_moco_segments(
        t_len,
        float(cfg.fps),
        core_s=float(cfg.moco_core_duration_s),
        buffer_s=float(cfg.moco_buffer_duration_s),
    )
    if not segments:
        raise ValueError(f"No segments for length {t_len}")

    if skeleton is not None:
        arr, ground_shift = apply_ground_offset_q(arr, skeleton, cfg)
    else:
        ground_shift = 0.0

    model_path = rajagopal_model_path()
    track_cfg = _track_config(cfg)
    names_ref = muscle_names()
    n_muscles = len(names_ref)
    blend_frames, _ = segment_frame_counts(
        float(cfg.fps),
        core_s=float(cfg.moco_stitch_blend_s),
        buffer_s=float(cfg.moco_buffer_duration_s),
    )

    core_activations: List[np.ndarray] = []
    core_grf: List[np.ndarray] = []
    segment_ok: List[bool] = []
    segment_details: List[Dict[str, Any]] = []

    with opensim_quiet(cfg.opensim_log_level):
        mapping = build_rajagopal_coord_mapping(model_path=model_path)
        moco_model_path = prepare_rajagopal_moco_track_model(
            work_dir,
            model_path=model_path,
            track_cfg=track_cfg,
        )

        for spec in segments:
            seg_dir = work_dir / f"segment_{spec.index:04d}"
            seg_dir.mkdir(parents=True, exist_ok=True)
            q_seg = arr[spec.solve_start : spec.solve_end]
            mesh_interval = _effective_mesh_interval(cfg, q_seg)
            activations, solve_ok, solve_meta, _, grf_seg = _solve_moco_track(
                q_seg,
                cfg=cfg,
                solve_dir=seg_dir,
                moco_model_path=moco_model_path,
                mapping=mapping,
                muscle_name_list=names_ref,
                mesh_interval=mesh_interval,
            )
            reserve_ok = bool(solve_meta.get("reserve_qc_pass", True))
            if bool(cfg.moco_fail_on_high_reserve) and not reserve_ok:
                solve_ok = False

            local_core_start = spec.core_start - spec.solve_start
            local_core_end = spec.core_end - spec.solve_start
            core_act = activations[local_core_start:local_core_end]
            core_grf_seg = grf_seg[local_core_start:local_core_end]

            if not solve_ok:
                core_act = np.full_like(core_act, np.nan, dtype=np.float32)
                core_grf_seg = np.full_like(core_grf_seg, np.nan, dtype=np.float32)

            core_activations.append(core_act.astype(np.float32))
            core_grf.append(core_grf_seg.astype(np.float32))
            segment_ok.append(bool(solve_ok))
            segment_details.append(
                {
                    "index": int(spec.index),
                    "solve_start": int(spec.solve_start),
                    "solve_end": int(spec.solve_end),
                    "core_start": int(spec.core_start),
                    "core_end": int(spec.core_end),
                    "solver_success": bool(solve_meta.get("solver_success", solve_ok)),
                    "success": bool(solve_ok),
                    "reserve_qc_pass": bool(solve_meta.get("reserve_qc_pass", True)),
                    "objective": solve_meta.get("objective"),
                    "solver_status": solve_meta.get("solver_status"),
                }
            )

    stitched_act = stitch_segment_values(
        t_len,
        segments,
        core_activations,
        blend_frames=blend_frames,
        stitch_seams=True,
    )
    stitched_grf = stitch_segment_values(
        t_len,
        segments,
        core_grf,
        blend_frames=blend_frames,
        stitch_seams=True,
    )
    validity_mask = stitch_segment_mask(t_len, segments, segment_ok)

    success_count = int(sum(1 for ok in segment_ok if ok))
    meta: Dict[str, Any] = {
        "activation_method": "moco_track",
        "moco_segmented": True,
        "ground_offset_m": float(ground_shift),
        "moco_segment_count": int(len(segments)),
        "moco_segment_success_count": success_count,
        "moco_segment_details": segment_details,
        "moco_segment_success_fraction": float(success_count / max(len(segments), 1)),
        "num_frames": t_len,
        "num_muscles": n_muscles,
        "fps": float(cfg.fps),
        "sim_grf": stitched_grf.astype(np.float32),
        "activation_validity_mask": validity_mask.astype(np.float32),
        "repaired_frame_count": 0,
    }
    return MuscleActivationResult(
        activations=stitched_act.astype(np.float32),
        muscle_names=tuple(names_ref),
        metadata=meta,
        forces=stitched_grf.astype(np.float32),
    )
