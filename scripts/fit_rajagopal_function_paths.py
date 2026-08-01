from __future__ import annotations
import argparse
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from common.paths import default_humanml3d_root, nimble_b3d_dir, repo_root
from datasets.nimble_dataset import read_q_segment
from datasets.splits import all_motion_ids
from nimble.muscle_activation import opensim_quiet
from nimble.rajagopal_coord_map import build_rajagopal_coord_mapping, write_coordinates_mot
from nimble.rajagopal_model import function_based_path_set_path, prepare_unlocked_rajagopal_base

def _sample_motion_ids(out_root: Path, sample_motions: int) -> list[str]:
    b3d_dir = nimble_b3d_dir(out_root)
    if b3d_dir.is_dir():
        ids = sorted(p.stem for p in b3d_dir.glob('*.b3d') if p.is_file())
        if ids:
            return ids[:sample_motions] if sample_motions > 0 else ids
    ids = all_motion_ids(out_root)
    return ids[:sample_motions] if sample_motions > 0 else ids[:50]

def fit_function_paths(*, out_root: Path, sample_motions: int=50, fps: float=20.0) -> dict:
    import opensim as osim
    ids = _sample_motion_ids(out_root, sample_motions)
    if not ids:
        raise RuntimeError('No motions available for path fitting')
    out_xml = function_based_path_set_path()
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix='sindyffuse_path_fit_'))
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
            fitter = osim.PolynomialPathFitter()
            fitter.setModel(osim.Model(str(base_model)))
            for mot in mot_paths:
                fitter.addCoordinateData(str(mot))
            fitter.setOutputPathSetFile(str(out_xml))
            fitter.run()
    finally:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)
    meta = {'output_xml': str(out_xml), 'sample_motions': len(ids), 'motion_ids': ids[:10]}
    (out_xml.parent / 'path_fit_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta

def main() -> None:
    parser = argparse.ArgumentParser(description='Job 2: fit Rajagopal function-based muscle paths from IK B3D q trajectories')
    parser.add_argument('--out_root', default=default_humanml3d_root())
    parser.add_argument('--sample_motions', type=int, default=50)
    parser.add_argument('--fps', type=float, default=20.0)
    args = parser.parse_args()
    out_root = Path(args.out_root).expanduser().resolve()
    result = fit_function_paths(out_root=out_root, sample_motions=int(args.sample_motions), fps=float(args.fps))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
