from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from common.paths import default_humanml3d_root, nimble_b3d_dir
from common.preprocess_runner import load_stage_manifest_index, manifest_path

def calibrate_ik_gates(*, out_root: Path, num_shards: int, percentile: float=95.0, sample_motions: int=0) -> dict:
    ik_index = load_stage_manifest_index(out_root, num_shards, stage='ik')
    mean_losses: list[float] = []
    max_losses: list[float] = []
    for row in ik_index.values():
        if row.get('status') not in {'ik_ok', 'ok'}:
            continue
        stats = row.get('ik_stats') or {}
        mean_v = stats.get('mean_fk_loss')
        max_v = stats.get('max_fk_loss')
        if mean_v is not None and np.isfinite(float(mean_v)):
            mean_losses.append(float(mean_v))
        if max_v is not None and np.isfinite(float(max_v)):
            max_losses.append(float(max_v))
        elif isinstance(stats.get('per_frame_fk_loss'), list):
            finite = [float(x) for x in stats['per_frame_fk_loss'] if np.isfinite(float(x))]
            if finite:
                max_losses.append(float(max(finite)))
    if sample_motions > 0:
        mean_losses = mean_losses[:sample_motions]
        max_losses = max_losses[:sample_motions]
    if not mean_losses or not max_losses:
        raise RuntimeError('No IK manifest stats available for calibration. Run preprocess_ik.py first.')
    cfg = {'max_mean_fk_loss': float(np.percentile(mean_losses, percentile)), 'max_max_fk_loss': float(np.percentile(max_losses, percentile)), 'percentile': float(percentile), 'num_motions': int(min(len(mean_losses), len(max_losses)))}
    out_path = out_root / 'ik_gate_config.json'
    out_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    return {'path': str(out_path), **cfg}

def main() -> None:
    parser = argparse.ArgumentParser(description='Calibrate IK FK gate thresholds from preprocess_ik manifests')
    parser.add_argument('--out_root', default=default_humanml3d_root())
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--percentile', type=float, default=95.0)
    parser.add_argument('--sample_motions', type=int, default=0)
    args = parser.parse_args()
    out_root = Path(args.out_root).expanduser().resolve()
    result = calibrate_ik_gates(out_root=out_root, num_shards=int(args.num_shards), percentile=float(args.percentile), sample_motions=int(args.sample_motions))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
