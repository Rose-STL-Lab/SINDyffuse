from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import hashlib
import time
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
# CasADi before any OpenSim import (OpenSim ships a conflicting libcasadi).
import casadi  # noqa: F401
from common.cpu import bootstrap_moco_compute_threads, configure_compute_threads, resolve_preprocess_parallelism
# OpenSim initializes OpenMP/MKL pools at import; configure before any opensim import.
_BOOTSTRAP_MOCO_THREADS = bootstrap_moco_compute_threads()
from common.paths import lai_cache_dir
from common.preprocess_runner import add_common_preprocess_args, load_stage_manifest_index, manifest_path, resolve_shard_motion_ids, run_preprocess_loop
from common.run_setup import apply_preprocess_job_env
from common.run_logging import add_run_log_cli_args, null_logger, run_log_session
from nimble.export import clear_export_caches, npz_has_activations, patch_npz_activations
from nimble.muscle_activation import add_muscle_activation_cli_args, configure_opensim_logging, muscle_activation_config_from_args, muscle_activation_config_from_dict, muscle_activation_config_to_dict
from nimble.opensim_log import opensim_quiet

def _task_artifacts(*, deep: bool=False) -> dict:
    from nimble.opensimad.paths import validate_opensimad_worker_artifacts, external_function_metadata_path, polynomial_cache_metadata_path
    validate_opensimad_worker_artifacts(load_library=True, deep=deep)
    # Preserve exact code and artifact provenance without tying tasks to git metadata.
    code_files = ['scripts/preprocess_moco.py', 'nimble/opensimad_track.py',
                  'nimble/opensimad/track_segment.py', 'nimble/opensimad/mint_settings.py',
                  'nimble/opensimad/label_processing.py',
                  'nimble/opensimad/vendor/opencap_ad/mainOpenSimAD.py',
                  'nimble/opensimad/vendor/opencap_ad/utilsOpenSimAD.py',
                  'nimble/opensimad/vendor/opencap_ad/functionCasADiOpenSimAD.py',
                  'nimble/opensimad/vendor/opencap_ad/polynomialsOpenSimAD.py']
    return {'worker_artifacts': {'external_function': json.loads(external_function_metadata_path().read_text()),
                                 'polynomial_cache': json.loads(polynomial_cache_metadata_path().read_text())},
            'code': {name: hashlib.sha256((_REPO / name).read_bytes()).hexdigest() for name in code_files}}

def run_motion_task(args, logger) -> None:
    from common.motion_tasks import prepare_tasks, load_tasks, task_motion, task_lock, outcome_path, atomic_json
    from common.preprocess_runner import load_manifest_rows
    from datasets.splits import all_motion_ids
    from datasets.lai_cache import read_motion_npz
    from common.paths import lai_cache_dir
    root = Path(args.out_root).expanduser().resolve()
    inputs = Path(args.hml_root).expanduser().resolve()
    directory = Path(args.motion_task_dir).expanduser().resolve()
    cfg = muscle_activation_config_from_args(args, fps=float(args.fps), mass_kg=float(args.mass_kg))
    cfg_dict = muscle_activation_config_to_dict(cfg)
    # Explicit budget avoids nested process/thread oversubscription. It is an
    # execution setting, not a numerical quality gate.
    budget = int(os.environ.get('MOCO_CPU_BUDGET', '0'))
    threads = int(os.environ.get('MOCO_NUM_THREADS', '1'))
    segments = int(getattr(cfg, 'moco_parallel_segments', 1) or 1)
    if budget > 0 and segments * threads > budget:
        raise ValueError(f'Segment/thread concurrency {segments} × {threads} exceeds CPU budget {budget}')
    artifacts = _task_artifacts(deep=args.motion_task_mode == 'prepare')
    artifacts['execution'] = {'solver_threads': threads, 'segment_workers': segments}
    if args.motion_task_mode == 'prepare':
        ids = all_motion_ids(inputs)
        if args.max_motions > 0:
            ids = ids[:args.max_motions]
        ik_rows = {}
        for path in sorted(root.glob('preprocess_ik_manifest*.jsonl')):
            ik_rows.update(load_manifest_rows(path))
        tasks = prepare_tasks(directory, motion_ids=ids, configuration=cfg_dict,
                              artifacts=artifacts, expected_tasks=args.expected_tasks, data_root=root,
                              ik_rows={mid: ik_rows[mid] for mid in ids if mid in ik_rows})
        logger.progress(f"Prepared {tasks['task_count']} motion tasks: {directory / 'tasks.json'}")
        return
    tasks = load_tasks(directory / 'tasks.json')
    if tasks['configuration'] != cfg_dict or tasks['artifacts'] != artifacts or tasks['data_root'] != str(root):
        raise ValueError('Motion task configuration/artifacts changed; choose a new task directory')
    raw_index = args.motion_task_index if args.motion_task_index is not None else os.environ.get('JOB_COMPLETION_INDEX', '')
    index = int(raw_index)
    mid = task_motion(tasks, index, args.expected_tasks)
    output = lai_cache_dir(root) / f'{mid}.npz'
    result_path = outcome_path(directory, index)
    with task_lock(directory / 'locks' / f'{index:06d}.lock'):
        if result_path.exists():
            old = json.loads(result_path.read_text())
            if old.get('task_set_id') != tasks['task_set_id'] or old.get('id') != mid:
                raise ValueError('Stale per-motion outcome')
            if old.get('status') in ('moco_failed', 'moco_skipped'):
                logger.progress(f'{mid}: recorded terminal dataset outcome; no deterministic retry')
                return
            try:
                reusable = (old.get('status') in ('ok', 'skipped') and npz_has_activations(output)
                            and old.get('result_sha256') == hashlib.sha256(output.read_bytes()).hexdigest())
            except Exception:
                reusable = False
            if reusable:
                logger.progress(f'{mid}: validated existing labels; task complete')
                return
        import_existing = bool(args.skip_existing) and not result_path.exists()
        # Read only IK manifests: production shard numbering is independent of tasks.
        ik = tasks.get('ik_rows', {}).get(mid, {})
        verbose = str(getattr(args, '_run_log_file', '') or '')
        start = time.monotonic()
        row = _process_one_moco((mid, str(root), import_existing, verbose, json.dumps(cfg_dict),
                                ik.get('status'), json.dumps(ik.get('ik_stats', {}))))
        row.update(task_index=index, task_set_id=tasks['task_set_id'], elapsed_seconds=time.monotonic() - start)
        if row['status'] in ('ok', 'skipped'):
            data = read_motion_npz(output)
            import numpy as np
            good = (data['muscle_activation_mask'] > .5) & np.isfinite(data['muscle_activations']).all(axis=1)
            if not good.any():
                row.update(status='error', error='Successful outcome has no finite valid activation frames')
            else:
                row['valid_frames'] = int(good.sum())
                row['result_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
                row['existing_labels_imported'] = row['status'] == 'skipped'
                row['label_processing_version'] = data.get('activation_diagnostics', {}).get('label_processing_version', 'legacy_unreported')
        atomic_json(result_path, row)
        logger.progress(json.dumps(row))
        if row['status'] == 'error':
            raise RuntimeError(row.get('error', 'Motion processing error'))
        if row['status'] in ('moco_failed', 'moco_skipped'):
            reason = (row.get('meta', {}).get('moco_failed_reason') or row.get('moco_skipped_reason') or '')
            expected_failure = ('precomputed default polynomial bounds' in reason or
                                'optimization did not converge' in reason or
                                (row['status'] == 'moco_skipped' and ik.get('status') == 'ik_failed'))
            if not expected_failure:
                # Unknown errors must not be permanently accepted as ordinary
                # dataset gaps. Leave a retryable outcome and fail this index.
                row.update(status='error', error=reason or 'Unclassified failed activation solve')
                atomic_json(result_path, row)
                raise RuntimeError(row['error'])
        # Nonconverged/out-of-domain samples are terminal dataset outcomes, not
        # infrastructure failures. Their labels remain masked/NaN.

def _process_one_moco(item: tuple) -> dict:
    sid, out_root_s, skip_existing, verbose_log_path, act_cfg_json, ik_status, ik_stats_json = item
    if verbose_log_path:
        os.environ['SINDYFFUSE_VERBOSE_LOG'] = str(verbose_log_path)
    out_npz = lai_cache_dir(Path(out_root_s)) / f'{sid}.npz'
    if skip_existing and out_npz.is_file():
        try:
            if npz_has_activations(out_npz):
                return {'id': sid, 'status': 'skipped', 'path': str(out_npz), 'skip_reason': 'existing opensimad npz'}
        except Exception:
            pass
    if not out_npz.is_file():
        return {'id': sid, 'status': 'moco_skipped', 'error': 'missing IK NPZ', 'moco_skipped_reason': 'missing NPZ'}
    act_cfg = muscle_activation_config_from_dict(json.loads(act_cfg_json))
    ik_stats = json.loads(ik_stats_json) if ik_stats_json else {}
    try:
        with opensim_quiet(act_cfg.opensim_log_level):
            stats, num_dofs, meta_strings, manifest_status = patch_npz_activations(
                out_npz, trial_name=sid, act_cfg=act_cfg, ik_manifest_status=ik_status, ik_stats=ik_stats
            )
    except Exception as exc:
        return {'id': sid, 'status': 'error', 'error': str(exc)}
    clear_export_caches()
    row = {'id': sid, 'status': manifest_status, 'path': str(out_npz), 'num_dofs': int(num_dofs), 'ik_stats': stats}
    if manifest_status == 'moco_skipped':
        row['moco_skipped_reason'] = meta_strings.get('moco_skipped_reason') or meta_strings.get('ik_gate_reason') or 'preflight gate'
    if manifest_status == 'moco_failed':
        row['coordinate_tracking_gate_reason'] = meta_strings.get('moco_failed_reason') or meta_strings.get('coordinate_tracking_gate_reason') or meta_strings.get('error') or 'opensimad failed'
    if meta_strings:
        row['meta'] = meta_strings
    return row

def run_preprocess_moco(args: argparse.Namespace, logger) -> None:
    if args.motion_task_mode != 'legacy':
        run_motion_task(args, logger)
        return
    from nimble.opensimad.paths import validate_opensimad_worker_artifacts
    artifacts = validate_opensimad_worker_artifacts(load_library=True, deep=False)
    logger.progress(
        'OpenSimAD preflight OK: '
        f"{artifacts['external_function']['library']} + precomputed polynomial cache"
    )
    ids, shard_index, num_shards, hml_root, out_root = resolve_shard_motion_ids(args)
    act_cfg = muscle_activation_config_from_args(args, fps=float(args.fps), mass_kg=float(args.mass_kg))
    act_cfg_json = json.dumps(muscle_activation_config_to_dict(act_cfg))
    ik_index = load_stage_manifest_index(out_root, num_shards, stage='ik')
    verbose = str(getattr(args, '_run_log_file', '') or '').strip()
    work = []
    for sid in ids:
        ik_row = ik_index.get(sid, {})
        ik_status = str(ik_row.get('status', '')) or None
        ik_stats_json = json.dumps(ik_row.get('ik_stats', {})) if ik_row.get('ik_stats') else ''
        work.append((sid, str(out_root), bool(args.skip_existing), verbose, act_cfg_json, ik_status, ik_stats_json))
    configure_opensim_logging(str(args.opensim_log_level))
    parallel_segments = int(getattr(args, 'moco_parallel_segments', act_cfg.moco_parallel_segments) or 1)
    motion_workers, moco_threads = resolve_preprocess_parallelism(int(args.num_workers), moco_parallel_motions=int(getattr(args, 'moco_parallel_motions', 1) or 1), moco_parallel_segments=parallel_segments, num_shards=num_shards)
    moco_threads = configure_compute_threads(moco_threads)
    logger.progress(f'OpenSimAD threads: bootstrap={_BOOTSTRAP_MOCO_THREADS} resolved={moco_threads} motion_workers={motion_workers}')
    manifest_file = manifest_path(out_root, shard_index, num_shards, stage='moco')
    ok, err, skip = run_preprocess_loop(work=work, process_one=_process_one_moco, manifest_file=manifest_file, motion_workers=motion_workers, moco_threads=moco_threads, ok_statuses={'ok'}, logger=logger)
    logger.progress(f'Done (opensimad): {ok} ok, {err} failed/skipped, {skip} skipped existing')
    if ok == 0 and skip == 0:
        sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(description='Job 5: OpenSimAD (MinT) muscle activations on Lai IK NPZ cache')
    add_common_preprocess_args(parser)
    parser.add_argument('--moco_parallel_motions', type=int, default=1)
    parser.add_argument('--motion_task_mode', choices=('legacy', 'prepare', 'process'), default=os.environ.get('MOTION_TASK_MODE', 'legacy'))
    parser.add_argument('--motion_task_dir', default=os.environ.get('MOTION_TASK_DIR', ''))
    parser.add_argument('--motion_task_index', type=int, default=None)
    parser.add_argument('--expected_tasks', type=int, default=int(os.environ.get('MOTION_TASK_COUNT', '0')))
    add_muscle_activation_cli_args(parser)
    add_run_log_cli_args(parser)
    args = parser.parse_args()
    if args.motion_task_mode != 'legacy' and (not args.motion_task_dir or args.expected_tasks <= 0):
        parser.error('Indexed motion tasks require --motion_task_dir and positive --expected_tasks')
    if args.motion_task_mode == 'process' and args.motion_task_index is None and not os.environ.get('JOB_COMPLETION_INDEX', '').isdigit():
        parser.error('process mode requires --motion_task_index or JOB_COMPLETION_INDEX')
    apply_preprocess_job_env(args)
    if not getattr(args, 'activation_method', None):
        args.activation_method = 'opensimad'
    else:
        from nimble.muscle_activation import normalize_activation_method
        args.activation_method = normalize_activation_method(str(args.activation_method))
    if not str(os.environ.get('SINDYFFUSE_RUN_LOG_ID', '')).strip() or str(os.environ.get('SINDYFFUSE_RUN_LOG_ID', '')).strip() == 'RUN_LOG_ID_PLACEHOLDER':
        shard = os.environ.get('JOB_COMPLETION_INDEX') or os.environ.get('PREPROCESS_SHARD_INDEX') or ''
        if str(shard).strip().isdigit():
            os.environ['SINDYFFUSE_RUN_LOG_ID'] = f'moco_shard_{int(shard):04d}'
        else:
            os.environ['SINDYFFUSE_RUN_LOG_ID'] = f'moco_{os.getpid()}'
    if args.no_run_log:
        run_preprocess_moco(args, null_logger())
        return
    with run_log_session(args.log_dir, script_name=Path(__file__).stem, argv=sys.argv) as (paths, logger):
        args._run_log_file = str(paths.log_file)
        logger.progress(f'log: {paths.latest_log}')
        logger.progress(f'activation_method={args.activation_method}')
        run_preprocess_moco(args, logger)

if __name__ == '__main__':
    main()
