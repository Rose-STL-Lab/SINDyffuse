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
from common.cpu import configure_compute_threads, detect_usable_cpus, resolve_k8s_shard
from common.paths import default_humanml3d_root, humanml3d_text_dir, nimble_b3d_dir
from common.preprocess_runner import load_stage_manifest_index
from datasets.nimble_dataset import read_q_segment
from datasets.splits import all_motion_ids, load_split_ids, shard_motion_ids
from nimble.muscle_activation import opensim_quiet
from nimble.rajagopal_coord_map import RajagopalCoordMapping, build_rajagopal_coord_mapping, write_coordinates_mot
from nimble.rajagopal_model import function_based_path_set_path, prepare_unlocked_rajagopal_base
DEFAULT_SAMPLE_MOTIONS = 200
DEFAULT_SAMPLE_SEED = 42
COORDINATE_TABLE_SUBSAMPLE_STRIDE = 5
MOTION_MANIFEST_NAME = 'path_fit_motion_ids.json'
CONVERT_DONE_PREFIX = 'path_fit_convert_done'
STAGING_SUBDIR = 'path_fit_mot'
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

def _sample_motion_ids(out_root: Path, sample_motions: int, *, num_shards: int=1, seed: int=DEFAULT_SAMPLE_SEED) -> list[str]:
    ik_index = load_stage_manifest_index(out_root, num_shards, stage='ik')
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

def motion_manifest_path(out_root: Path) -> Path:
    return out_root / MOTION_MANIFEST_NAME

def staging_dir(out_root: Path, staging_dir: Path | None=None) -> Path:
    return staging_dir if staging_dir is not None else out_root / STAGING_SUBDIR

def convert_done_marker_path(out_root: Path, shard_index: int) -> Path:
    return out_root / f'{CONVERT_DONE_PREFIX}.{int(shard_index):04d}'

def _resolve_ik_num_shards(num_shards: int | None) -> int:
    if num_shards is not None:
        return max(1, int(num_shards))
    raw = os.environ.get('PREPROCESS_NUM_SHARDS', '').strip()
    return max(1, int(raw)) if raw.isdigit() else 1

def _resolve_convert_num_shards(num_shards: int | None) -> int:
    if num_shards is not None:
        return max(1, int(num_shards))
    raw = os.environ.get('PATH_FIT_NUM_SHARDS', '').strip()
    return max(1, int(raw)) if raw.isdigit() else 1

def _load_motion_manifest(out_root: Path) -> dict:
    path = motion_manifest_path(out_root)
    if not path.is_file():
        raise FileNotFoundError(f'Motion manifest missing: {path}. Run --phase prepare first.')
    return json.loads(path.read_text(encoding='utf-8'))

def _write_motion_manifest(out_root: Path, *, motion_ids: list[str], sample_motions: int, sample_seed: int, ik_num_shards: int) -> Path:
    path = motion_manifest_path(out_root)
    payload = {'motion_ids': motion_ids, 'sample_motions': int(sample_motions), 'sample_seed': int(sample_seed), 'ik_num_shards': int(ik_num_shards), 'sampling': 'split_and_caption_stratified_systematic'}
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return path

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

def _resolve_num_threads(num_threads: int | None) -> int | None:
    if num_threads is not None:
        return max(1, int(num_threads))
    raw = os.environ.get('PATH_FIT_NUM_THREADS', '').strip()
    if not raw:
        return None
    return max(1, int(raw))

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

def _run_path_fitter(*, base_model: Path, mot_paths: list[Path], num_threads: int | None, work_dir: Path) -> Path:
    import opensim as osim
    resolved_threads = _resolve_num_threads(num_threads)
    if resolved_threads is not None:
        configure_compute_threads(resolved_threads)
    coordinates = _subsample_coordinate_table(_merge_coordinate_tables(list(mot_paths)), stride=COORDINATE_TABLE_SUBSAMPLE_STRIDE)
    fitter = osim.PolynomialPathFitter()
    if resolved_threads is not None:
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

def _cleanup_staging(out_root: Path, *, motion_ids: list[str], num_convert_shards: int) -> None:
    staging = staging_dir(out_root)
    if staging.is_dir():
        shutil.rmtree(staging, ignore_errors=True)
    for shard_index in range(num_convert_shards):
        convert_done_marker_path(out_root, shard_index).unlink(missing_ok=True)

def phase_prepare(*, out_root: Path, sample_motions: int, sample_seed: int, ik_num_shards: int) -> dict:
    ids = _sample_motion_ids(out_root, sample_motions, num_shards=ik_num_shards, seed=sample_seed)
    if not ids:
        raise RuntimeError('No motions available for path fitting')
    manifest_path = _write_motion_manifest(out_root, motion_ids=ids, sample_motions=sample_motions, sample_seed=sample_seed, ik_num_shards=ik_num_shards)
    return {'phase': 'prepare', 'manifest_path': str(manifest_path), 'motion_count': len(ids), 'motion_ids': ids[:10]}

def phase_convert(*, out_root: Path, fps: float, convert_num_shards: int, shard_index: int | None, staging_dir_arg: Path | None) -> dict:
    manifest = _load_motion_manifest(out_root)
    motion_ids: list[str] = list(manifest['motion_ids'])
    shard_i, num_shards = resolve_k8s_shard(num_shards=convert_num_shards, shard_index=shard_index)
    if num_shards != convert_num_shards:
        raise ValueError(f'convert_num_shards={convert_num_shards} must match resolved num_shards={num_shards}')
    shard_ids = shard_motion_ids(motion_ids, shard_i, num_shards)
    staging = staging_dir(out_root, staging_dir_arg)
    staging.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix='sindyffuse_path_fit_convert_'))
    converted: list[str] = []
    try:
        with opensim_quiet('Off'):
            base_model = prepare_unlocked_rajagopal_base(work_dir)
            mapping = build_rajagopal_coord_mapping(model_path=base_model)
            for sid in shard_ids:
                mot = _convert_motion_to_mot(sid=sid, out_root=out_root, staging=staging, fps=fps, mapping=mapping)
                if mot is not None:
                    converted.append(sid)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    convert_done_marker_path(out_root, shard_i).write_text(json.dumps({'shard_index': shard_i, 'num_shards': num_shards, 'motion_ids': converted}, indent=2), encoding='utf-8')
    return {'phase': 'convert', 'shard_index': shard_i, 'num_shards': num_shards, 'converted': len(converted), 'motion_ids': converted[:10]}

def phase_fit(*, out_root: Path, fps: float, convert_num_shards: int, num_threads: int | None, staging_dir_arg: Path | None, cleanup: bool=True) -> dict:
    manifest = _load_motion_manifest(out_root)
    motion_ids: list[str] = list(manifest['motion_ids'])
    sample_seed = int(manifest.get('sample_seed', DEFAULT_SAMPLE_SEED))
    staging = staging_dir(out_root, staging_dir_arg)
    mot_paths: list[Path] = []
    for sid in motion_ids:
        mot = staging / f'{sid}.mot'
        if mot.is_file():
            mot_paths.append(mot)
    if not mot_paths:
        raise RuntimeError(f'No staged .mot files found under {staging}')
    out_xml = function_based_path_set_path()
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix='sindyffuse_path_fit_'))
    resolved_threads = _resolve_num_threads(num_threads)
    try:
        with opensim_quiet('Off'):
            base_model = prepare_unlocked_rajagopal_base(work_dir)
            generated = _run_path_fitter(base_model=base_model, mot_paths=mot_paths, num_threads=num_threads, work_dir=work_dir)
            shutil.copy2(generated, out_xml)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    meta = {'phase': 'fit', 'output_xml': str(out_xml), 'sample_motions': len(motion_ids), 'sample_seed': sample_seed, 'sampling': manifest.get('sampling', 'split_and_caption_stratified_systematic'), 'motion_ids': motion_ids[:10], 'num_threads': resolved_threads if resolved_threads is not None else 'default', 'mot_files': len(mot_paths)}
    (out_xml.parent / 'path_fit_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    if cleanup:
        _cleanup_staging(out_root, motion_ids=motion_ids, num_convert_shards=convert_num_shards)
    return meta

def phase_all(*, out_root: Path, sample_motions: int, fps: float, ik_num_shards: int, sample_seed: int, num_threads: int | None, num_workers: int) -> dict:
    ids = _sample_motion_ids(out_root, sample_motions, num_shards=ik_num_shards, seed=sample_seed)
    if not ids:
        raise RuntimeError('No motions available for path fitting')
    out_xml = function_based_path_set_path()
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix='sindyffuse_path_fit_'))
    resolved_threads = _resolve_num_threads(num_threads)
    workers = detect_usable_cpus() if int(num_workers) <= 0 else max(1, int(num_workers))
    try:
        with opensim_quiet('Off'):
            base_model = prepare_unlocked_rajagopal_base(work_dir)
            mapping = build_rajagopal_coord_mapping(model_path=base_model)
            staging = work_dir / 'mot'
            staging.mkdir(parents=True, exist_ok=True)
            mot_paths = _convert_motions_parallel(ids=ids, out_root=out_root, staging=staging, fps=fps, mapping=mapping, base_model=base_model, num_workers=workers)
            if not mot_paths:
                raise RuntimeError('No B3D coordinate tables found for path fitting')
            generated = _run_path_fitter(base_model=base_model, mot_paths=mot_paths, num_threads=num_threads, work_dir=work_dir)
            shutil.copy2(generated, out_xml)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    meta = {'phase': 'all', 'output_xml': str(out_xml), 'sample_motions': len(ids), 'sample_seed': int(sample_seed), 'sampling': 'split_and_caption_stratified_systematic', 'motion_ids': ids[:10], 'num_threads': resolved_threads if resolved_threads is not None else 'default', 'num_workers': workers}
    (out_xml.parent / 'path_fit_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta

def main() -> None:
    parser = argparse.ArgumentParser(description='Job 2: fit Rajagopal function-based muscle paths from IK B3D q trajectories')
    parser.add_argument('--out_root', default=default_humanml3d_root())
    parser.add_argument('--phase', choices=('all', 'prepare', 'convert', 'fit'), default='all', help='Pipeline phase (default: all — local super-node).')
    parser.add_argument('--sample_motions', type=int, default=int(os.environ.get('PATH_FIT_SAMPLE_MOTIONS', DEFAULT_SAMPLE_MOTIONS) or DEFAULT_SAMPLE_MOTIONS))
    parser.add_argument('--sample_seed', type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument('--fps', type=float, default=20.0)
    parser.add_argument('--num_shards', type=int, default=None, help='IK manifest shards for prepare (PREPROCESS_NUM_SHARDS) or convert/fit shards (PATH_FIT_NUM_SHARDS).')
    parser.add_argument('--shard_index', type=int, default=-1, help='Convert shard index (default: JOB_COMPLETION_INDEX in Indexed Job).')
    parser.add_argument('--num_threads', type=int, default=None, help='PolynomialPathFitter parallel threads (default: OpenSim default, overridable via PATH_FIT_NUM_THREADS).')
    parser.add_argument('--num_workers', type=int, default=0, help='B3D→.mot worker processes for --phase all (default: detect_usable_cpus()).')
    parser.add_argument('--staging_dir', default='', help='Override staging directory for convert/fit (default: {out_root}/path_fit_mot).')
    parser.add_argument('--no_cleanup', action='store_true', help='Keep staging .mot files after --phase fit.')
    args = parser.parse_args()
    out_root = Path(args.out_root).expanduser().resolve()
    staging_dir_arg = Path(args.staging_dir).expanduser().resolve() if str(args.staging_dir).strip() else None
    shard_index = int(args.shard_index) if int(args.shard_index) >= 0 else None
    phase = str(args.phase)
    if phase == 'prepare':
        ik_num_shards = _resolve_ik_num_shards(args.num_shards)
        result = phase_prepare(out_root=out_root, sample_motions=int(args.sample_motions), sample_seed=int(args.sample_seed), ik_num_shards=ik_num_shards)
    elif phase == 'convert':
        convert_num_shards = _resolve_convert_num_shards(args.num_shards)
        result = phase_convert(out_root=out_root, fps=float(args.fps), convert_num_shards=convert_num_shards, shard_index=shard_index, staging_dir_arg=staging_dir_arg)
    elif phase == 'fit':
        convert_num_shards = _resolve_convert_num_shards(args.num_shards)
        result = phase_fit(out_root=out_root, fps=float(args.fps), convert_num_shards=convert_num_shards, num_threads=args.num_threads, staging_dir_arg=staging_dir_arg, cleanup=not bool(args.no_cleanup))
    else:
        ik_num_shards = _resolve_ik_num_shards(args.num_shards if args.num_shards is not None else _resolve_ik_num_shards(None))
        result = phase_all(out_root=out_root, sample_motions=int(args.sample_motions), fps=float(args.fps), ik_num_shards=ik_num_shards, sample_seed=int(args.sample_seed), num_threads=args.num_threads, num_workers=int(args.num_workers))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
