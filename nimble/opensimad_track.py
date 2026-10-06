from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import sys
from typing import Any, Dict, List
import numpy as np
import casadi  # noqa: F401
from nimble.moco_segment import plan_moco_segments, segment_frame_counts, stitch_segment_mask
from nimble.muscle_activation import MuscleActivationConfig, MuscleActivationResult, muscle_names, opensim_quiet
from nimble.opensimad.mint_settings import MINT_PARALLEL_SEGMENTS
from nimble.opensimad.track_segment import solve_opensimad_segment
from nimble.opensim_ik import apply_ground_offset_lai_q
from nimble.opensimad.label_processing import stitch_solve_windows, pool_tracking, set_grf_validity, LABEL_PROCESSING_VERSION

def _solve_one_segment_job(args: tuple) -> tuple:
    spec_index, q_seg, cfg_dict, solve_dir_s, mesh_interval = args
    from nimble.muscle_activation import muscle_activation_config_from_dict
    cfg = muscle_activation_config_from_dict(cfg_dict)
    activations, solve_ok, solve_meta, grf = solve_opensimad_segment(
        q_seg, cfg=cfg, solve_dir=Path(solve_dir_s), mesh_interval=mesh_interval)
    return (int(spec_index), activations, bool(solve_ok), solve_meta, grf)

def solve_one_opensimad_segment_isolated(job: tuple) -> tuple:
    """Solve one segment in a spawned process so native memory dies with it."""
    ctx = get_context('spawn')
    kwargs = {'max_workers': 1, 'mp_context': ctx}
    if sys.version_info >= (3, 11):
        kwargs['max_tasks_per_child'] = 1
    with ProcessPoolExecutor(**kwargs) as executor:
        return executor.submit(_solve_one_segment_job, job).result()

def run_opensimad_segmented(q: np.ndarray, *, cfg: MuscleActivationConfig, work_dir: Path) -> MuscleActivationResult:
    from nimble.muscle_activation import muscle_activation_config_to_dict
    arr = np.asarray(q, dtype=np.float64)
    t_len = int(arr.shape[0])
    segments = plan_moco_segments(t_len, float(cfg.fps), core_s=float(cfg.moco_core_duration_s), buffer_s=float(cfg.moco_buffer_duration_s))
    if not segments:
        raise ValueError(f'No segments for length {t_len}')
    arr, ground_shift = apply_ground_offset_lai_q(arr, sphere_offset_y_m=float(cfg.moco_contact_sphere_offset_y_m))
    names_ref = muscle_names()
    n_muscles = len(names_ref)
    blend_frames, _ = segment_frame_counts(float(cfg.fps), core_s=float(cfg.moco_stitch_blend_s), buffer_s=float(cfg.moco_buffer_duration_s))
    mesh_interval = float(cfg.mesh_interval) if cfg.mesh_interval is not None else 0.02
    parallel = max(1, int(cfg.moco_parallel_segments or MINT_PARALLEL_SEGMENTS))
    cfg_dict = muscle_activation_config_to_dict(cfg)

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
                idx, act, ok, meta, grf = solve_one_opensimad_segment_isolated(job)
                results_by_index[idx] = (act, ok, meta, grf)
        else:
            # Cap concurrent in-flight futures to max_workers so we do not
            # pickle/queue every segment NLP up front.
            workers = min(parallel, len(jobs))
            ctx = get_context('spawn')
            pool_kwargs = {'max_workers': workers, 'mp_context': ctx}
            if sys.version_info >= (3, 11):
                pool_kwargs['max_tasks_per_child'] = 1
            with ProcessPoolExecutor(**pool_kwargs) as ex:
                pending: set = set()
                job_iter = iter(jobs)

                def _submit_next() -> bool:
                    try:
                        job = next(job_iter)
                    except StopIteration:
                        return False
                    pending.add(ex.submit(_solve_one_segment_job, job))
                    return True

                for _ in range(workers):
                    if not _submit_next():
                        break
                while pending:
                    fut = next(as_completed(pending))
                    pending.remove(fut)
                    idx, act, ok, meta, grf = fut.result()
                    results_by_index[idx] = (act, ok, meta, grf)
                    _submit_next()

    solve_activations = []
    solve_grfs = []
    tracking = []
    segment_ok: List[bool] = []
    segment_details: List[Dict[str, Any]] = []
    for spec in segments:
        activations, solve_ok, solve_meta, grf_seg = results_by_index[spec.index]
        segment_ok.append(bool(solve_ok))
        solve_activations.append(activations)
        solve_grfs.append(grf_seg)
        if solve_ok and solve_meta.get('coordinate_tracking'):
            tracking.append(solve_meta['coordinate_tracking'])
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
        for key in ('coordinate_tracking', 'ipopt_return_status', 'ipopt_iterations', 'grf_valid_frames'):
            if key in solve_meta:
                detail[key] = solve_meta[key]
        segment_details.append(detail)

    stitched_act = stitch_solve_windows(t_len, segments, solve_activations, segment_ok, blend_frames=blend_frames)
    stitched_grf = set_grf_validity(stitch_solve_windows(t_len, segments, solve_grfs, segment_ok, blend_frames=blend_frames))
    validity_mask = stitch_segment_mask(t_len, segments, segment_ok)
    success_count = int(sum((1 for ok in segment_ok if ok)))
    pooled_tracking = pool_tracking(tracking)
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
        'max_translational_coord_rmse_m': pooled_tracking['max_translational_rmse_m'],
        'max_rotational_coord_rmse_deg': pooled_tracking['max_rotational_rmse_deg'],
        'label_processing_version': LABEL_PROCESSING_VERSION,
        'grf_torque_convention': 'resultant free moment at COP',
        'tracking_available': bool(tracking),
        'opensim_model': 'LaiUhlrich2022',
    }
    return MuscleActivationResult(activations=stitched_act.astype(np.float32), muscle_names=tuple(names_ref), metadata=meta, forces=stitched_grf.astype(np.float32))

def run_opensimad_track(q: np.ndarray, *, cfg: MuscleActivationConfig, work_dir: Path) -> MuscleActivationResult:
    return run_opensimad_segmented(np.asarray(q, dtype=np.float64), cfg=cfg, work_dir=work_dir)
