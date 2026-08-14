from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import numpy as np
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from common.cpu import configure_compute_threads, detect_usable_cpus
from common.paths import default_humanml3d_root, humanml3d_text_dir, nimble_b3d_dir
# OpenSim initializes OpenMP/MKL pools at import; configure before nimble imports it.
_path_fit_threads = os.environ.get('PATH_FIT_NUM_THREADS', '').strip()
if _path_fit_threads.isdigit():
    configure_compute_threads(int(_path_fit_threads))
from common.preprocess_runner import load_stage_manifest_index
from datasets.nimble_dataset import read_q_segment
from datasets.splits import all_motion_ids, load_split_ids
from nimble.muscle_activation import opensim_quiet
from nimble.rajagopal_coord_map import RajagopalCoordMapping, build_rajagopal_coord_mapping, write_coordinates_mot
from nimble.rajagopal_model import function_based_path_set_path, prepare_welded_unlocked_rajagopal_base
DEFAULT_SAMPLE_MOTIONS = 200
DEFAULT_SAMPLE_SEED = 42
COORDINATE_TABLE_SUBSAMPLE_STRIDE = 5
_SPLIT_NAMES = ('train', 'val', 'test')

def _caption_key(text_dir: Path, sid: str) -> str:
    path = text_dir / f'{sid}.txt'
    if not path.is_file():
        return '__missing__'
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line:
            continue
        caption = line.split('#', 1)[0].strip().lower()
        return caption or '__empty__'
    return '__empty__'

def _proportional_quotas(counts: list[int], total: int) -> list[int]:
    if total <= 0 or not counts:
        return [0] * len(counts)
    total_count = sum(counts)
    if total_count <= 0:
        return [0] * len(counts)
    if total >= total_count:
        return list(counts)
    raw = [total * c / total_count for c in counts]
    floors = [int(x) for x in raw]
    remainder = total - sum(floors)
    order = sorted(range(len(counts)), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in order[:remainder]:
        floors[i] += 1
    return floors

def _systematic_pick(sorted_ids: list[str], take: int) -> list[str]:
    if take <= 0:
        return []
    if take >= len(sorted_ids):
        return list(sorted_ids)
    idx = np.linspace(0, len(sorted_ids) - 1, take, dtype=int)
    return [sorted_ids[int(i)] for i in idx]

def _sample_from_pool(ids: list[str], *, text_dir: Path, quota: int, rng: np.random.Generator) -> list[str]:
    if quota <= 0 or not ids:
        return []
    groups: dict[str, list[str]] = {}
    for sid in ids:
        groups.setdefault(_caption_key(text_dir, sid), []).append(sid)
    keys = sorted(groups.keys())
    rng.shuffle(keys)
    group_lists = [sorted(groups[k]) for k in keys]
    takes = _proportional_quotas([len(g) for g in group_lists], quota)
    picked: list[str] = []
    for group, take in zip(group_lists, takes):
        picked.extend(_systematic_pick(group, take))
    if len(picked) < quota:
        remaining = sorted(set(ids) - set(picked))
        picked.extend(_systematic_pick(remaining, quota - len(picked)))
    return picked[:quota]

def _diverse_sample_motion_ids(out_root: Path, ok_ids: list[str], sample_motions: int, *, seed: int) -> list[str]:
    if sample_motions <= 0 or len(ok_ids) <= sample_motions:
        return ok_ids if sample_motions <= 0 else ok_ids[:sample_motions]
    rng = np.random.default_rng(int(seed))
    ok_set = set(ok_ids)
    text_dir = humanml3d_text_dir(out_root)
    split_pools: list[list[str]] = []
    split_sizes: list[int] = []
    for split in _SPLIT_NAMES:
        try:
            pool = sorted(sid for sid in load_split_ids(out_root, split) if sid in ok_set)
        except FileNotFoundError:
            pool = []
        if pool:
            split_pools.append(pool)
            split_sizes.append(len(pool))
    if not split_pools:
        return _sample_from_pool(sorted(ok_ids), text_dir=text_dir, quota=sample_motions, rng=rng)
    split_quotas = _proportional_quotas(split_sizes, sample_motions)
    picked: list[str] = []
    for pool, quota in zip(split_pools, split_quotas):
        picked.extend(_sample_from_pool(pool, text_dir=text_dir, quota=quota, rng=rng))
    if len(picked) < sample_motions:
        remaining = sorted(ok_set - set(picked))
        picked.extend(_sample_from_pool(remaining, text_dir=text_dir, quota=sample_motions - len(picked), rng=rng))
    return picked[:sample_motions]

def _sample_motion_ids(out_root: Path, sample_motions: int, *, ik_num_shards: int, seed: int=DEFAULT_SAMPLE_SEED) -> list[str]:
    ik_index = load_stage_manifest_index(out_root, ik_num_shards, stage='ik')
    ok_ids = sorted(mid for mid, row in ik_index.items() if row.get('status') == 'ik_ok')
    if ok_ids:
        return _diverse_sample_motion_ids(out_root, ok_ids, sample_motions, seed=seed)
    b3d_dir = nimble_b3d_dir(out_root)
    if b3d_dir.is_dir():
        ids = sorted(p.stem for p in b3d_dir.glob('*.b3d') if p.is_file())
        if ids:
            return _diverse_sample_motion_ids(out_root, ids, sample_motions, seed=seed)
    ids = all_motion_ids(out_root)
    return _diverse_sample_motion_ids(out_root, ids, sample_motions, seed=seed)

def _resolve_ik_num_shards(num_shards: int | None) -> int:
    if num_shards is not None:
        return max(1, int(num_shards))
    raw = os.environ.get('PREPROCESS_NUM_SHARDS', '').strip()
    return max(1, int(raw)) if raw.isdigit() else 1

def _copy_mot_paths_for_merge(mot_paths: list[Path], work_dir: Path) -> list[Path]:
    merge_dir = work_dir / 'mot_merge'
    merge_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for index, mot in enumerate(mot_paths):
        dest = merge_dir / f'{index:04d}_{mot.name}'
        shutil.copy2(mot, dest)
        copied.append(dest)
    return copied

def _merge_coordinate_tables(mot_paths: list[Path]):
    import opensim as osim
    combined = osim.TimeSeriesTable(str(mot_paths[0]))
    mot_paths[0].unlink(missing_ok=True)
    times = combined.getIndependentColumn()
    time_offset = float(times[-1]) + 0.05
    for mot in mot_paths[1:]:
        table = osim.TimeSeriesTable(str(mot))
        mot_times = table.getIndependentColumn()
        for i in range(table.getNumRows()):
            combined.appendRow(time_offset + float(mot_times[i]), table.getRowAtIndex(i))
        mot.unlink(missing_ok=True)
        times = combined.getIndependentColumn()
        time_offset = float(times[-1]) + 0.05
    return combined

def _subsample_coordinate_table(table, *, stride: int):
    if stride <= 1:
        return table
    times = table.getIndependentColumn()
    for i in range(len(times)):
        if i % stride != 0:
            table.removeRow(times[i])
    return table

def _resolve_num_threads(num_threads: int | None) -> int:
    if num_threads is not None:
        return max(1, int(num_threads))
    raw = os.environ.get('PATH_FIT_NUM_THREADS', '').strip()
    if raw.isdigit():
        return max(1, int(raw))
    return detect_usable_cpus()

def _resolve_num_workers(num_workers: int) -> int:
    if int(num_workers) > 0:
        return max(1, int(num_workers))
    raw = os.environ.get('PATH_FIT_NUM_WORKERS', '').strip()
    if raw.isdigit():
        return max(1, int(raw))
    return detect_usable_cpus()

def _convert_motion_to_mot(*, sid: str, out_root: Path, staging: Path, fps: float, mapping: RajagopalCoordMapping) -> Path | None:
    b3d = nimble_b3d_dir(out_root) / f'{sid}.b3d'
    if not b3d.is_file():
        return None
    q = read_q_segment(str(b3d))
    mot = staging / f'{sid}.mot'
    write_coordinates_mot(q, mot, fps=float(fps), mapping=mapping)
    return mot

def _convert_worker(args: tuple[str, str, str, float, str]) -> tuple[str, str | None]:
    sid, out_root_s, staging_s, fps, model_path_s = args
    mapping = build_rajagopal_coord_mapping(model_path=model_path_s)
    mot = _convert_motion_to_mot(sid=sid, out_root=Path(out_root_s), staging=Path(staging_s), fps=float(fps), mapping=mapping)
    return (sid, str(mot) if mot is not None else None)

def _convert_motions_parallel(*, ids: list[str], out_root: Path, staging: Path, fps: float, mapping: RajagopalCoordMapping, base_model: Path, num_workers: int) -> list[Path]:
    workers = max(1, int(num_workers))
    if workers <= 1 or len(ids) <= 1:
        mot_paths: list[Path] = []
        for sid in ids:
            mot = _convert_motion_to_mot(sid=sid, out_root=out_root, staging=staging, fps=fps, mapping=mapping)
            if mot is not None:
                mot_paths.append(mot)
        return mot_paths
    ctx = get_context('spawn')
    tasks = [(sid, str(out_root), str(staging), float(fps), str(base_model)) for sid in ids]
    mot_paths = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futures = [ex.submit(_convert_worker, task) for task in tasks]
        for fut in as_completed(futures):
            sid, mot_s = fut.result()
            if mot_s is not None:
                mot_paths.append(Path(mot_s))
    order = {sid: i for i, sid in enumerate(ids)}
    mot_paths.sort(key=lambda p: order.get(p.stem, len(order)))
    return mot_paths

def _run_path_fitter(*, base_model: Path, mot_paths: list[Path], num_threads: int, work_dir: Path) -> Path:
    import opensim as osim
    resolved_threads = configure_compute_threads(num_threads)
    merge_inputs = _copy_mot_paths_for_merge(mot_paths, work_dir)
    coordinates = _subsample_coordinate_table(_merge_coordinate_tables(merge_inputs), stride=COORDINATE_TABLE_SUBSAMPLE_STRIDE)
    fitter = osim.PolynomialPathFitter()
    fitter.setNumParallelThreads(resolved_threads)
    fitter.setModel(osim.ModelProcessor(str(base_model)))
    fitter.setCoordinateValues(osim.TableProcessor(coordinates))
    fit_out_dir = work_dir / 'path_fit_out'
    fit_out_dir.mkdir(parents=True, exist_ok=True)
    fitter.setOutputDirectory(str(fit_out_dir))
    fitter.run()
    model_name = osim.Model(str(base_model)).getName()
    generated = fit_out_dir / f'{model_name}_FunctionBasedPathSet.xml'
    if not generated.is_file():
        raise RuntimeError(f'PolynomialPathFitter did not write expected output: {generated}')
    return generated

def fit_function_paths(*, out_root: Path, sample_motions: int=DEFAULT_SAMPLE_MOTIONS, fps: float=20.0, ik_num_shards: int=1, seed: int=DEFAULT_SAMPLE_SEED, num_threads: int | None=None, num_workers: int=0) -> dict:
    ids = _sample_motion_ids(out_root, sample_motions, ik_num_shards=ik_num_shards, seed=seed)
    if not ids:
        raise RuntimeError('No motions available for path fitting')
    out_xml = function_based_path_set_path()
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix='sindyffuse_path_fit_'))
    resolved_threads = configure_compute_threads(_resolve_num_threads(num_threads))
    workers = _resolve_num_workers(num_workers)
    try:
        with opensim_quiet('Off'):
            # Fit on welded-MTP Rajagopal so FunctionBasedPathSet does not reference mtp_angle_*
            # (required for OpenSimAD / MinT-aligned toe welding).
            base_model = prepare_welded_unlocked_rajagopal_base(work_dir)
            mapping = build_rajagopal_coord_mapping(model_path=base_model)
            staging = work_dir / 'mot'
            staging.mkdir(parents=True, exist_ok=True)
            mot_paths = _convert_motions_parallel(ids=ids, out_root=out_root, staging=staging, fps=fps, mapping=mapping, base_model=base_model, num_workers=workers)
            if not mot_paths:
                raise RuntimeError('No B3D coordinate tables found for path fitting')
            generated = _run_path_fitter(base_model=base_model, mot_paths=mot_paths, num_threads=resolved_threads, work_dir=work_dir)
            shutil.copy2(generated, out_xml)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    meta = {
        'output_xml': str(out_xml),
        'sample_motions': len(ids),
        'sample_seed': int(seed),
        'sampling': 'split_and_caption_stratified_systematic',
        'motion_ids': ids[:10],
        'num_threads': resolved_threads,
        'num_workers': workers,
        'mtp_welded': True,
        'path_fit_model': 'rajagopal_unlocked_mtp_welded',
    }
    (out_xml.parent / 'path_fit_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta

def main() -> None:
    parser = argparse.ArgumentParser(description='Job 2: fit Rajagopal function-based muscle paths from IK B3D q trajectories')
    parser.add_argument('--out_root', default=default_humanml3d_root())
    parser.add_argument('--sample_motions', type=int, default=int(os.environ.get('PATH_FIT_SAMPLE_MOTIONS', DEFAULT_SAMPLE_MOTIONS) or DEFAULT_SAMPLE_MOTIONS))
    parser.add_argument('--sample_seed', type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument('--fps', type=float, default=20.0)
    parser.add_argument('--num_shards', type=int, default=None, help='IK manifest shard count (PREPROCESS_NUM_SHARDS).')
    parser.add_argument('--num_threads', type=int, default=None, help='PolynomialPathFitter parallel threads (default: PATH_FIT_NUM_THREADS or cgroup CPU count).')
    parser.add_argument('--num_workers', type=int, default=0, help='B3D→.mot worker processes (default: PATH_FIT_NUM_WORKERS or cgroup CPU count).')
    args = parser.parse_args()
    out_root = Path(args.out_root).expanduser().resolve()
    ik_num_shards = _resolve_ik_num_shards(args.num_shards)
    result = fit_function_paths(out_root=out_root, sample_motions=int(args.sample_motions), fps=float(args.fps), ik_num_shards=ik_num_shards, seed=int(args.sample_seed), num_threads=args.num_threads, num_workers=int(args.num_workers))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
