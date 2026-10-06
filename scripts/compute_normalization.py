from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from common.paths import NIMBLE_B3D_SUBDIR, cleanup_preprocess_manifests, default_humanml3d_root, nimble_b3d_dir
from common.run_setup import env_int, require_nimble_b3d
from common.run_logging import RunLogger, add_run_log_cli_args, null_logger, run_log_session
from datasets.nimble_dataset import compute_nimble_normalization_stats

def _shard_manifest_path(out_root: Path, shard_index: int, *, stage: str='moco') -> Path:
    return out_root / f'preprocess_{stage}_manifest.{int(shard_index):04d}.jsonl'

def _wait_for_shards(out_root: Path, num_shards: int, *, timeout_hours: float, poll_seconds: float, logger: RunLogger) -> None:
    deadline = time.monotonic() + float(timeout_hours) * 3600.0
    while True:
        missing = [i for i in range(num_shards) if not _shard_manifest_path(out_root, i, stage='moco').is_file()]
        if not missing:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Timed out after {timeout_hours}h waiting for shard manifest(s): {missing}')
        logger.verbose(f'Waiting for {len(missing)} shard manifest(s); missing indices: {missing[:8]}' + ('...' if len(missing) > 8 else ''))
        time.sleep(float(poll_seconds))

def _load_manifest_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows

def compute_normalization(args: argparse.Namespace, logger: RunLogger | None=None) -> dict:
    log = logger or null_logger()
    default_root = default_humanml3d_root()
    out_root = Path(getattr(args, 'out_root', default_root) or default_root).expanduser().resolve()
    num_shards = int(getattr(args, 'num_shards', 1) or 1)
    task_directory = str(getattr(args, 'motion_task_dir', '') or '')
    task_rows = None
    tasks = None
    if task_directory:
        from common.motion_tasks import load_outcomes
        tasks, task_rows = load_outcomes(Path(task_directory).expanduser().resolve())
        if tasks['data_root'] != str(out_root):
            raise ValueError('Motion task dataset root does not match normalization root')
    if not task_directory and num_shards <= 1:
        raise ValueError('--num_shards must be > 1 (use preprocess_ik.py / preprocess_moco.py directly when not sharded)')
    if not task_directory and bool(getattr(args, 'wait', False)):
        _wait_for_shards(out_root, num_shards, timeout_hours=float(getattr(args, 'timeout_hours', 48.0)), poll_seconds=float(getattr(args, 'poll_seconds', 30.0)), logger=log)
    missing = [] if task_directory else [i for i in range(num_shards) if not _shard_manifest_path(out_root, i, stage='moco').is_file()]
    if missing:
        raise FileNotFoundError(f'Missing shard manifest(s) under {out_root}: {missing}. Re-run missing shards or use --wait.')
    rows_by_id: dict[str, dict] = {}
    ok = err = skip = 0
    num_dofs_ref: int | None = None
    shard_meta: list[dict] = []
    for shard_index in range(1 if task_directory else num_shards):
        shard_path = _shard_manifest_path(out_root, shard_index, stage='moco')
        # Restart-safe manifests may contain an earlier failed row followed by
        # a successful retry. The final row for a motion is authoritative.
        shard_rows_by_id: dict[str, dict] = {}
        for row in (task_rows if task_rows is not None else _load_manifest_rows(shard_path)):
            mid = str(row.get('id', ''))
            if mid:
                shard_rows_by_id[mid] = row
        shard_rows = list(shard_rows_by_id.values())
        shard_ok = shard_err = shard_skip = 0
        for row in shard_rows:
            mid = str(row.get('id', ''))
            if not mid:
                continue
            if mid in rows_by_id:
                log.warn(f'Duplicate motion id {mid} in shard {shard_index}; keeping first')
                continue
            rows_by_id[mid] = row
            status = row.get('status')
            if status == 'ok':
                ok += 1
                shard_ok += 1
                nd = int(row.get('num_dofs', 0))
                if num_dofs_ref is None:
                    num_dofs_ref = nd
            elif status == 'skipped':
                skip += 1
                shard_skip += 1
            else:
                err += 1
                shard_err += 1
        shard_meta.append({'shard_index': shard_index, 'manifest_path': str(shard_path), 'motions_ok': shard_ok, 'motions_error': shard_err, 'motions_skipped': shard_skip})
    merged_path = out_root / 'preprocess_manifest.jsonl'
    sorted_ids = sorted(rows_by_id.keys())
    with merged_path.open('w', encoding='utf-8') as mf:
        for mid in sorted_ids:
            mf.write(json.dumps(rows_by_id[mid], default=str) + '\n')
    b3d_cache = nimble_b3d_dir(out_root)
    meta = {'out_root': str(out_root), 'b3d_subdir': NIMBLE_B3D_SUBDIR, 'nimble_b3d_dir': str(b3d_cache), 'num_dofs': num_dofs_ref, 'motions_ok': ok, 'motions_error': err, 'motions_skipped': skip, 'num_shards': num_shards, 'shard_manifests': shard_meta, 'manifest_path': str(merged_path), 'normalization_computed': True}
    if tasks is not None:
        meta.update(motion_task_set_id=tasks['task_set_id'], motion_task_count=tasks['task_count'],
                    motion_task_dir=task_directory, num_shards=0, shard_manifests=[])
    (out_root / 'preprocess_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    log.progress(f'Merged {len(sorted_ids)} motion(s): {ok} ok, {err} failed, {skip} skipped')
    if ok == 0 and skip == 0:
        raise RuntimeError('No successful or skipped motions after merge')
    stats = compute_nimble_normalization_stats(out_root)
    log.progress(f"Wrote normalization stats: {stats['mean_path']}")
    removed = [] if task_directory else cleanup_preprocess_manifests(out_root)
    if removed:
        log.verbose(f'Removed {len(removed)} temporary preprocess manifest file(s)')
    return meta

def main() -> None:
    parser = argparse.ArgumentParser(description='Merge shard preprocess manifests and compute Nimble q Mean.npy / Std.npy')
    default_root = default_humanml3d_root()
    parser.add_argument('--out_root', default=default_root, help='Dataset root containing lai_cache/ and shard manifests')
    parser.add_argument('--num_shards', type=int, default=0, help='Number of preprocess shards to merge (default: PREPROCESS_NUM_SHARDS env or 1)')
    parser.add_argument('--motion_task_dir', default='', help='Merge completed one-motion task outcomes instead of legacy shards')
    parser.add_argument('--wait', action='store_true', help='Poll until all shard manifests exist before merging')
    parser.add_argument('--timeout_hours', type=float, default=48.0, help='Max wait time when --wait is set (default 48)')
    parser.add_argument('--poll_seconds', type=float, default=30.0, help='Poll interval when --wait is set (default 30)')
    add_run_log_cli_args(parser)
    args = parser.parse_args()
    if int(args.num_shards) <= 0:
        args.num_shards = env_int('PREPROCESS_NUM_SHARDS', 1)
    out_root = Path(str(args.out_root)).expanduser().resolve()
    require_nimble_b3d(out_root)
    if args.no_run_log:
        compute_normalization(args, null_logger())
        return
    script_name = Path(__file__).stem
    with run_log_session(args.log_dir, script_name=script_name, argv=sys.argv) as (paths, logger):
        logger.progress(f'log: {paths.latest_log}')
        compute_normalization(args, logger)
if __name__ == '__main__':
    main()