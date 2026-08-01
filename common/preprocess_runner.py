from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import get_context
try:
    from concurrent.futures.process import BrokenProcessPool
except ImportError:
    BrokenProcessPool = BrokenExecutor
from pathlib import Path
from typing import Callable
import numpy as np
from common.cpu import configure_compute_threads, resolve_k8s_shard, resolve_preprocess_parallelism
from common.paths import NIMBLE_B3D_SUBDIR, default_humanml3d_root, humanml3d_text_dir, nimble_b3d_dir
from common.run_logging import DualTqdm, RunLogger, dual_tqdm, null_logger
from datasets.splits import all_motion_ids, shard_motion_ids

def symlink_metadata(hml_root: Path, out_root: Path) -> None:
    if hml_root.resolve() == out_root.resolve():
        return
    src_texts = humanml3d_text_dir(hml_root)
    if src_texts.is_dir() and src_texts != hml_root.resolve():
        dst = out_root / 'texts'
        if not dst.exists():
            if sys.platform != 'win32':
                try:
                    os.symlink(src_texts, dst, target_is_directory=True)
                    return
                except OSError:
                    pass
            shutil.copytree(src_texts, dst)
    for name in ('train.txt', 'val.txt', 'test.txt'):
        src = hml_root / name
        dst = out_root / name
        if src.is_file() and (not dst.exists()):
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)

def manifest_path(out_root: Path, shard_index: int, num_shards: int, *, stage: str) -> Path:
    prefix = f'preprocess_{stage}_manifest'
    if int(num_shards) > 1:
        return out_root / f'{prefix}.{int(shard_index):04d}.jsonl'
    return out_root / f'{prefix}.jsonl'

def load_manifest_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        mid = str(row.get('id', ''))
        if mid:
            rows[mid] = row
    return rows

def load_stage_manifest_index(out_root: Path, num_shards: int, *, stage: str) -> dict[str, dict]:
    if num_shards <= 1:
        return load_manifest_rows(manifest_path(out_root, 0, 1, stage=stage))
    merged: dict[str, dict] = {}
    for shard_index in range(num_shards):
        merged.update(load_manifest_rows(manifest_path(out_root, shard_index, num_shards, stage=stage)))
    return merged

def print_motion_progress(row: dict, *, logger: RunLogger) -> None:
    mid = str(row.get('id', ''))
    status = row.get('status')
    logger.verbose(f'=== motion {mid} status={status} ===')
    if status == 'skipped':
        return
    if status not in {'ok', 'ik_ok'}:
        err = row.get('error', row.get('moco_skipped_reason', 'failed'))
        logger.warn(f'{mid}: {err}')
        return
    ik = row.get('ik_stats') or {}
    seg_ok = ik.get('moco_segment_success_count')
    if seg_ok is not None and status == 'ok' and float(seg_ok) <= 0:
        logger.warn(f'{mid}: manifest ok but segment_success_count=0')

def run_preprocess_loop(*, work: list[tuple], process_one: Callable[[tuple], dict], manifest_file: Path, motion_workers: int, moco_threads: int, ok_statuses: set[str], logger: RunLogger, isolate_motion_process: bool=False) -> tuple[int, int, int]:
    ok = err = skip = 0
    with manifest_file.open('w', encoding='utf-8') as mf:

        def write_row(row: dict) -> None:
            mf.write(json.dumps(row, default=str) + '\n')
            mf.flush()
            os.fsync(mf.fileno())

        def record(row: dict, pbar: DualTqdm | None) -> None:
            nonlocal ok, err, skip
            print_motion_progress(row, logger=logger)
            status = row.get('status')
            if status in ok_statuses:
                ok += 1
            elif status == 'skipped':
                skip += 1
            else:
                err += 1
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({'last': str(row.get('id', '')), 'status': str(status)}, refresh=False)

        pending = list(work)
        while pending:
            batch = pending
            pending = []
            pbar = dual_tqdm(total=len(batch), desc='preprocess', unit='motion', logger=logger)
            try:
                if motion_workers <= 1:
                    for item in batch:
                        row = process_one(item)
                        write_row(row)
                        record(row, pbar)
                else:
                    pool_broken = False
                    finished: set[str] = set()
                    pool_kwargs: dict = {'max_workers': motion_workers}
                    if sys.version_info >= (3, 11):
                        pool_kwargs['max_tasks_per_child'] = 1
                    with ProcessPoolExecutor(**pool_kwargs) as ex:
                        futs = {ex.submit(process_one, item): item for item in batch}
                        for fut in as_completed(futs):
                            item = futs[fut]
                            try:
                                row = fut.result()
                            except (BrokenProcessPool, BrokenExecutor) as exc:
                                pool_broken = True
                                row = {'id': item[0], 'status': 'error', 'error': f'worker crashed: {exc}'}
                            write_row(row)
                            record(row, pbar)
                            finished.add(str(item[0]))
                            if pool_broken:
                                break
                    if pool_broken:
                        pending = [it for it in batch if str(it[0]) not in finished] + pending
                        motion_workers = 1
                        logger.warn(f'Retrying {len(pending)} motion(s) sequentially after pool crash')
            finally:
                pbar.close()
    return (ok, err, skip)

def resolve_shard_motion_ids(args: argparse.Namespace) -> tuple[list[str], int, int, Path, Path]:
    default_root = default_humanml3d_root()
    hml_root = Path(getattr(args, 'hml_root', default_root) or default_root).expanduser().resolve()
    out_root = Path(getattr(args, 'out_root', default_root) or default_root).expanduser().resolve()
    ids = all_motion_ids(hml_root)
    if int(getattr(args, 'max_motions', 0) or 0) > 0:
        ids = ids[: int(args.max_motions)]
    shard_index, num_shards = resolve_k8s_shard(num_shards=int(getattr(args, 'num_shards', 1) or 1), shard_index=int(getattr(args, 'shard_index', -1)) if int(getattr(args, 'shard_index', -1)) >= 0 else None)
    if num_shards > 1:
        ids = shard_motion_ids(ids, shard_index, num_shards)
    symlink_metadata(hml_root, out_root)
    nimble_b3d_dir(out_root).mkdir(parents=True, exist_ok=True)
    return (ids, shard_index, num_shards, hml_root, out_root)

def add_common_preprocess_args(parser: argparse.ArgumentParser) -> None:
    default_root = default_humanml3d_root()
    parser.add_argument('--hml_root', default=default_root)
    parser.add_argument('--out_root', default=default_root)
    parser.add_argument('--fps', type=float, default=20.0)
    parser.add_argument('--mass_kg', type=float, default=70.0)
    parser.add_argument('--height_m', type=float, default=1.75)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--max_motions', type=int, default=0)
    parser.add_argument('--joint_source', choices=('auto', 'joints', 'new_joints'), default='auto')
    parser.add_argument('--joints_root', default='')
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--shard_index', type=int, default=-1)
    parser.add_argument('--skip_normalization', action='store_true')
    parser.add_argument('--ik_gate_config', default='')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--opensim_log_level', default='Off')
