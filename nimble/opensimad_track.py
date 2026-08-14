from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
from nimble.moco_segment import apply_ground_offset_q, plan_moco_segments, segment_frame_counts, stitch_segment_mask, stitch_segment_values
from nimble.muscle_activation import MuscleActivationConfig, MuscleActivationResult, muscle_names, opensim_quiet
from nimble.opensimad.mint_settings import MINT_PARALLEL_SEGMENTS
from nimble.opensimad.track_segment import solve_opensimad_segment

def _solve_one_segment_job(args: tuple) -> tuple:
    spec_index, q_seg, cfg_dict, solve_dir_s, mesh_interval = args
    from nimble.muscle_activation import muscle_activation_config_from_dict
    cfg = muscle_activation_config_from_dict(cfg_dict)
    activations, solve_ok, solve_meta, grf = solve_opensimad_segment(
        q_seg, cfg=cfg, solve_dir=Path(solve_dir_s), mesh_interval=mesh_interval)
    return (int(spec_index), activations, bool(solve_ok), solve_meta, grf)

def run_opensimad_segmented(q: np.ndarray, *, cfg: MuscleActivationConfig, work_dir: Path, skeleton: Any | None=None) -> MuscleActivationResult:
    from nimble.muscle_activation import muscle_activation_config_to_dict
    arr = np.asarray(q, dtype=np.float64)
    t_len = int(arr.shape[0])
    segments = plan_moco_segments(t_len, float(cfg.fps), core_s=float(cfg.moco_core_duration_s), buffer_s=float(cfg.moco_buffer_duration_s))
    if not segments:
        raise ValueError(f'No segments for length {t_len}')
    if skeleton is not None:
        arr, ground_shift = apply_ground_offset_q(arr, skeleton, cfg)
    else:
        ground_shift = 0.0
    names_ref = muscle_names()
    n_muscles = len(names_ref)
    blend_frames, _ = segment_frame_counts(float(cfg.fps), core_s=float(cfg.moco_stitch_blend_s), buffer_s=float(cfg.moco_buffer_duration_s))
    mesh_interval = float(cfg.mesh_interval) if cfg.mesh_interval is not None else 0.02
    parallel = max(1, int(cfg.moco_parallel_segments or MINT_PARALLEL_SEGMENTS))
    cfg_dict = muscle_activation_config_to_dict(cfg)

    # Pre-create segment dirs and jobs.
    jobs = []
    for spec in segments:
        seg_dir = work_dir / f'segment_{spec.index:04d}'
        seg_dir.mkdir(parents=True, exist_ok=True)
        q_seg = arr[spec.solve_start:spec.solve_end]
        jobs.append((spec.index, q_seg, cfg_dict, str(seg_dir), mesh_interval))

    results_by_index: Dict[int, tuple] = {}
    with opensim_quiet(cfg.opensim_log_level):
        if parallel <= 1 or len(jobs) <= 1:
            for job in jobs:
                idx, act, ok, meta, grf = _solve_one_segment_job(job)
                results_by_index[idx] = (act, ok, meta, grf)
        else:
            workers = min(parallel, len(jobs))
            ctx = get_context('spawn')
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                futs = [ex.submit(_solve_one_segment_job, job) for job in jobs]
                for fut in as_completed(futs):
                    idx, act, ok, meta, grf = fut.result()
                    results_by_index[idx] = (act, ok, meta, grf)

    core_activations: List[np.ndarray] = []
    core_grf: List[np.ndarray] = []
    segment_ok: List[bool] = []
    segment_details: List[Dict[str, Any]] = []
    for spec in segments:
        activations, solve_ok, solve_meta, grf_seg = results_by_index[spec.index]
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
        detail = {
            'index': int(spec.index),
            'solve_start': int(spec.solve_start),
            'solve_end': int(spec.solve_end),
            'core_start': int(spec.core_start),
            'core_end': int(spec.core_end),
            'solver_success': bool(solve_meta.get('solver_success', solve_ok)),
            'success': bool(solve_ok),
            'solver_status': solve_meta.get('solver_status'),
        }
        if solve_meta.get('error'):
            detail['error'] = str(solve_meta['error'])
        segment_details.append(detail)

    stitched_act = stitch_segment_values(t_len, segments, core_activations, blend_frames=blend_frames, stitch_seams=True)
    stitched_grf = stitch_segment_values(t_len, segments, core_grf, blend_frames=blend_frames, stitch_seams=True)
    validity_mask = stitch_segment_mask(t_len, segments, segment_ok)
    success_count = int(sum((1 for ok in segment_ok if ok)))
    # OpenSimAD path: no separate Moco coordinate table extraction; leave tracking empty (gap policy).
    pooled_tracking: Dict[str, Any] = {}
    meta: Dict[str, Any] = {
        'activation_method': 'opensimad',
        'moco_segmented': True,
        'ground_offset_m': float(ground_shift),
        'moco_segment_count': int(len(segments)),
        'moco_segment_success_count': success_count,
        'moco_segment_details': segment_details,
        'moco_segment_success_fraction': float(success_count / max(len(segments), 1)),
        'num_frames': t_len,
        'num_muscles': n_muscles,
        'fps': float(cfg.fps),
        'sim_grf': stitched_grf.astype(np.float32),
        'activation_validity_mask': validity_mask.astype(np.float32),
        'repaired_frame_count': 0,
        'coordinate_tracking': pooled_tracking,
        'max_translational_coord_rmse_m': 0.0,
        'max_rotational_coord_rmse_deg': 0.0,
    }
    return MuscleActivationResult(activations=stitched_act.astype(np.float32), muscle_names=tuple(names_ref), metadata=meta, forces=stitched_grf.astype(np.float32))

def run_opensimad_track(q: np.ndarray, *, cfg: MuscleActivationConfig, work_dir: Path) -> MuscleActivationResult:
    from nimble.physics import load_model
    sk = load_model().skeleton
    return run_opensimad_segmented(np.asarray(q, dtype=np.float64), cfg=cfg, work_dir=work_dir, skeleton=sk)
