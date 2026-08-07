from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from common.cpu import configure_compute_threads, resolve_preprocess_parallelism
from common.paths import nimble_b3d_dir
from common.preprocess_runner import add_common_preprocess_args, load_stage_manifest_index, manifest_path, resolve_shard_motion_ids, run_preprocess_loop
from common.run_setup import apply_preprocess_job_env
from common.run_logging import add_run_log_cli_args, null_logger, run_log_session
from nimble.export import clear_export_caches, patch_b3d_moco
from nimble.muscle_activation import add_muscle_activation_cli_args, configure_opensim_logging, muscle_activation_config_from_args, muscle_activation_config_from_dict, muscle_activation_config_to_dict, opensim_quiet

def _process_one_moco(item: tuple) -> dict:
    sid, out_root_s, skip_existing, verbose_log_path, act_cfg_json, ik_status, ik_stats_json = item
    if verbose_log_path:
        os.environ['SINDYFFUSE_VERBOSE_LOG'] = str(verbose_log_path)
    out_b3d = nimble_b3d_dir(Path(out_root_s)) / f'{sid}.b3d'
    if skip_existing and out_b3d.is_file():
        subj_ok = False
        try:
            import nimblephysics as nimble
            from nimble.b3d_io import b3d_has_muscle_activations
            subj = nimble.biomechanics.SubjectOnDisk(str(out_b3d))
            subj_ok = b3d_has_muscle_activations(subj)
        except Exception:
            subj_ok = False
        if subj_ok:
            return {'id': sid, 'status': 'skipped', 'path': str(out_b3d), 'skip_reason': 'existing moco b3d'}
    if not out_b3d.is_file():
        return {'id': sid, 'status': 'moco_skipped', 'error': 'missing IK B3D', 'moco_skipped_reason': 'missing B3D'}
    act_cfg = muscle_activation_config_from_dict(json.loads(act_cfg_json))
    ik_stats = json.loads(ik_stats_json) if ik_stats_json else {}
    try:
        with opensim_quiet(act_cfg.opensim_log_level):
            stats, num_dofs, meta_strings, manifest_status = patch_b3d_moco(out_b3d, trial_name=sid, act_cfg=act_cfg, ik_manifest_status=ik_status, ik_stats=ik_stats)
    except Exception as exc:
        return {'id': sid, 'status': 'error', 'error': str(exc)}
    clear_export_caches()
    row = {'id': sid, 'status': manifest_status, 'path': str(out_b3d), 'num_dofs': int(num_dofs), 'ik_stats': stats}
    if manifest_status == 'moco_skipped':
        row['moco_skipped_reason'] = meta_strings.get('moco_skipped_reason') or meta_strings.get('ik_gate_reason') or 'preflight gate'
    if manifest_status == 'moco_failed':
        row['coordinate_tracking_gate_reason'] = meta_strings.get('coordinate_tracking_gate_reason') or meta_strings.get('error') or 'moco failed'
    if meta_strings:
        row['meta'] = meta_strings
    return row

def run_preprocess_moco(args: argparse.Namespace, logger) -> None:
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
    configure_compute_threads(moco_threads)
    manifest_file = manifest_path(out_root, shard_index, num_shards, stage='moco')
    ok, err, skip = run_preprocess_loop(work=work, process_one=_process_one_moco, manifest_file=manifest_file, motion_workers=motion_workers, moco_threads=moco_threads, ok_statuses={'ok'}, logger=logger)
    logger.progress(f'Done (moco): {ok} ok, {err} failed/skipped, {skip} skipped existing')
    if ok == 0 and skip == 0:
        sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(description='Job 3: MocoTrack muscle activations on IK B3D cache')
    add_common_preprocess_args(parser)
    parser.add_argument('--moco_parallel_motions', type=int, default=1)
    add_muscle_activation_cli_args(parser)
    add_run_log_cli_args(parser)
    args = parser.parse_args()
    apply_preprocess_job_env(args)
    args.activation_method = 'moco_track'
    if args.no_run_log:
        run_preprocess_moco(args, null_logger())
        return
    with run_log_session(args.log_dir, script_name=Path(__file__).stem, argv=sys.argv) as (paths, logger):
        args._run_log_file = str(paths.log_file)
        logger.progress(f'log: {paths.latest_log}')
        run_preprocess_moco(args, logger)

if __name__ == '__main__':
    main()
