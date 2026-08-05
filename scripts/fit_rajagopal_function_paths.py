from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from common.paths import default_humanml3d_root, humanml3d_text_dir, nimble_b3d_dir
from common.preprocess_runner import load_stage_manifest_index
from datasets.nimble_dataset import read_q_segment
from datasets.splits import all_motion_ids, load_split_ids
from nimble.muscle_activation import opensim_quiet
from nimble.rajagopal_coord_map import build_rajagopal_coord_mapping, write_coordinates_mot
from nimble.rajagopal_model import function_based_path_set_path, prepare_unlocked_rajagopal_base
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

def fit_function_paths(*, out_root: Path, sample_motions: int=DEFAULT_SAMPLE_MOTIONS, fps: float=20.0, num_shards: int=1, seed: int=DEFAULT_SAMPLE_SEED, num_threads: int | None=None) -> dict:
    import opensim as osim
    ids = _sample_motion_ids(out_root, sample_motions, num_shards=num_shards, seed=seed)
    if not ids:
        raise RuntimeError('No motions available for path fitting')
    out_xml = function_based_path_set_path()
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix='sindyffuse_path_fit_'))
    resolved_threads = _resolve_num_threads(num_threads)
    try:
        with opensim_quiet('Off'):
            base_model = prepare_unlocked_rajagopal_base(work_dir)
            mapping = build_rajagopal_coord_mapping(model_path=base_model)
            mot_paths: list[Path] = []
            for sid in ids:
                b3d = nimble_b3d_dir(out_root) / f'{sid}.b3d'
                if not b3d.is_file():
                    continue
                q = read_q_segment(str(b3d))
                mot = work_dir / f'{sid}.mot'
                write_coordinates_mot(q, mot, fps=float(fps), mapping=mapping)
                mot_paths.append(mot)
            if not mot_paths:
                raise RuntimeError('No B3D coordinate tables found for path fitting')
            coordinates = _subsample_coordinate_table(_merge_coordinate_tables(mot_paths), stride=COORDINATE_TABLE_SUBSAMPLE_STRIDE)
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
            shutil.copy2(generated, out_xml)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    meta = {'output_xml': str(out_xml), 'sample_motions': len(ids), 'sample_seed': int(seed), 'sampling': 'split_and_caption_stratified_systematic', 'motion_ids': ids[:10], 'num_threads': resolved_threads if resolved_threads is not None else 'default'}
    (out_xml.parent / 'path_fit_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta

def main() -> None:
    parser = argparse.ArgumentParser(description='Job 2: fit Rajagopal function-based muscle paths from IK B3D q trajectories')
    parser.add_argument('--out_root', default=default_humanml3d_root())
    parser.add_argument('--sample_motions', type=int, default=DEFAULT_SAMPLE_MOTIONS)
    parser.add_argument('--sample_seed', type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument('--fps', type=float, default=20.0)
    parser.add_argument('--num_shards', type=int, default=int(os.environ.get('PREPROCESS_NUM_SHARDS', '1') or 1))
    parser.add_argument('--num_threads', type=int, default=None, help='PolynomialPathFitter parallel threads (default: OpenSim default, overridable via PATH_FIT_NUM_THREADS).')
    args = parser.parse_args()
    out_root = Path(args.out_root).expanduser().resolve()
    result = fit_function_paths(out_root=out_root, sample_motions=int(args.sample_motions), fps=float(args.fps), num_shards=int(args.num_shards), seed=int(args.sample_seed), num_threads=args.num_threads)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
