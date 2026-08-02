from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from common.preprocess_runner import add_common_preprocess_args, manifest_path, resolve_shard_motion_ids, run_preprocess_loop
from common.run_setup import apply_preprocess_job_env
from common.run_logging import add_run_log_cli_args, null_logger, run_log_session
from datasets.hml3d_joints import default_joints_root, load_hml3d_joint_positions
from datasets.nimble_dataset import compute_nimble_normalization_stats
from nimble.export import clear_export_caches, export_ik_to_b3d
from nimble.muscle_activation import configure_opensim_logging, opensim_quiet
from common.paths import nimble_b3d_dir

def _process_one_ik(item: tuple) -> dict:
    sid, hml_root_s, out_root_s, skip_existing, joint_source, joints_root_s, verbose_log_path, fps, mass_kg, height_m = item
    if verbose_log_path:
        os.environ['SINDYFFUSE_VERBOSE_LOG'] = str(verbose_log_path)
    out_b3d = nimble_b3d_dir(Path(out_root_s)) / f'{sid}.b3d'
    if skip_existing and out_b3d.is_file():
        return {'id': sid, 'status': 'skipped', 'path': str(out_b3d)}
    try:
        joints, _ = load_hml3d_joint_positions(Path(hml_root_s), sid, joint_source=joint_source, joints_root=Path(joints_root_s) if joints_root_s else None)
    except FileNotFoundError:
        return {'id': sid, 'status': 'error', 'error': 'missing or invalid motion'}
    try:
        with opensim_quiet('Off'):
            stats, num_dofs, meta_strings, manifest_status = export_ik_to_b3d(joints, out_b3d, trial_name=sid, fps=float(fps), mass_kg=float(mass_kg), height_m=float(height_m))
    except Exception as exc:
        return {'id': sid, 'status': 'error', 'error': str(exc)}
    clear_export_caches()
    row = {'id': sid, 'status': manifest_status, 'path': str(out_b3d), 'num_dofs': int(num_dofs), 'ik_stats': stats}
    if manifest_status == 'ik_failed':
        reason = meta_strings.get('ik_gate_reason')
        if reason:
            row['ik_gate_reason'] = reason
    if meta_strings:
        row['meta'] = meta_strings
    return row

def run_preprocess_ik(args: argparse.Namespace, logger) -> None:
    ids, shard_index, num_shards, hml_root, out_root = resolve_shard_motion_ids(args)
    joints_root_s = str(getattr(args, 'joints_root', '') or '').strip()
    if not joints_root_s:
        jr = default_joints_root(hml_root)
        joints_root_s = str(jr) if jr else ''
    verbose = str(getattr(args, '_run_log_file', '') or '').strip()
    work = [(sid, str(hml_root), str(out_root), bool(args.skip_existing), str(args.joint_source), joints_root_s, verbose, float(args.fps), float(args.mass_kg), float(args.height_m)) for sid in ids]
    configure_opensim_logging(str(args.opensim_log_level))
    from common.cpu import resolve_preprocess_parallelism
    motion_workers, _ = resolve_preprocess_parallelism(int(args.num_workers), activation_method='none', skip_muscle_activation=True, num_shards=num_shards)
    manifest_file = manifest_path(out_root, shard_index, num_shards, stage='ik')
    ok, err, skip = run_preprocess_loop(work=work, process_one=_process_one_ik, manifest_file=manifest_file, motion_workers=motion_workers, moco_threads=1, ok_statuses={'ik_ok'}, logger=logger)
    logger.progress(f'Done (ik): {ok} ik_ok, {err} failed, {skip} skipped')
    if ok == 0 and skip == 0:
        sys.exit(1)
    if not bool(getattr(args, 'skip_normalization', False)) and num_shards <= 1:
        compute_nimble_normalization_stats(out_root)

def main() -> None:
    parser = argparse.ArgumentParser(description='Job 1: HumanML3D joints → IK B3D cache')
    add_common_preprocess_args(parser)
    add_run_log_cli_args(parser)
    args = parser.parse_args()
    apply_preprocess_job_env(args)
    if args.no_run_log:
        run_preprocess_ik(args, null_logger())
        return
    with run_log_session(args.log_dir, script_name=Path(__file__).stem, argv=sys.argv) as (paths, logger):
        args._run_log_file = str(paths.log_file)
        logger.progress(f'log: {paths.latest_log}')
        run_preprocess_ik(args, logger)

if __name__ == '__main__':
    main()
