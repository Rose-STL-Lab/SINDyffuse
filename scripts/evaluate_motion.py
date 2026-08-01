from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from common.io import load_json
from common.paths import nimble_b3d_dir, resolve_data_root, resolve_repo_path
from common.run_logging import RunLogger, add_run_log_cli_args, run_logged_main
from datasets.nimble_dataset import read_q_frames
from datasets.splits import load_split_ids
from eval.aggregate import sem, summarize_with_bootstrap
from eval.biomechanical import compute_biomechanical_metrics
from eval.protocol import BOOTSTRAP_REPLICATES, FPS, NUM_SAMPLES_PER_CAPTION
from eval.text_alignment import text_alignment_bundle

def _load_norm_stats(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    cache = nimble_b3d_dir(data_root)
    mean = np.load(cache / 'Mean.npy').astype(np.float32)
    std = np.load(cache / 'Std.npy').astype(np.float32)
    return (mean, std)

def _load_generation_q(path: Path) -> np.ndarray:
    data = np.load(path)
    if 'motion' not in data:
        raise ValueError(f"{path} missing 'motion' array")
    motion = np.asarray(data['motion'], dtype=np.float32)
    if motion.ndim == 3:
        motion = motion[0]
    if motion.ndim != 2:
        raise ValueError(f'Expected motion [T, ndof] in {path}, got {motion.shape}')
    return motion

def _discover_generation_files(generations_dir: Path, motion_id: str) -> List[Path]:
    direct = sorted(generations_dir.glob(f'{motion_id}_*.npz'))
    if direct:
        return direct
    sub = generations_dir / motion_id
    if sub.is_dir():
        return sorted(sub.glob('*.npz'))
    single = generations_dir / f'{motion_id}.npz'
    if single.is_file():
        return [single]
    return []

def _load_reference_q(data_root: Path, motion_id: str) -> np.ndarray:
    b3d_path = nimble_b3d_dir(data_root) / f'{motion_id}.b3d'
    if not b3d_path.is_file():
        raise FileNotFoundError(f'Missing reference B3D: {b3d_path}')
    import nimblephysics as nimble
    subj = nimble.biomechanics.SubjectOnDisk(str(b3d_path))
    trial = 0
    n = int(subj.getTrialLength(trial))
    return read_q_frames(subj, trial, 0, n)

def evaluate_biomechanical_split(*, data_root: Path, generations_dir: Path, motion_ids: List[str], fps: float, mass_kg: float, height_thresh_m: float, speed_thresh_mps: float) -> Dict[str, Any]:
    mean, std = _load_norm_stats(data_root)
    per_caption: List[Dict[str, float]] = []
    for motion_id in motion_ids:
        gen_files = _discover_generation_files(generations_dir, motion_id)
        if not gen_files:
            continue
        try:
            q_ref = _load_reference_q(data_root, motion_id)
        except FileNotFoundError:
            continue
        sample_metrics: List[Dict[str, float]] = []
        for gen_path in gen_files:
            q = _load_generation_q(gen_path)
            t = min(q.shape[0], q_ref.shape[0])
            m = compute_biomechanical_metrics(q[:t], fps=float(fps), q_ref=q_ref[:t], mean=mean, std=std, mass_kg=float(mass_kg), height_thresh_m=float(height_thresh_m), speed_thresh_mps=float(speed_thresh_mps))
            sample_metrics.append(m)
        if not sample_metrics:
            continue
        avg = {k: float(np.mean([row[k] for row in sample_metrics if k in row])) for k in sample_metrics[0]}
        per_caption.append(avg)
    keys = list(per_caption[0].keys()) if per_caption else []
    bio_summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        vals = [row[key] for row in per_caption]
        bio_summary[key] = {'mean': float(np.mean(vals)), 'sem': sem(vals)}
    return {'per_caption': per_caption, 'summary': bio_summary}

def evaluate_text_from_embeddings(motion_emb: np.ndarray, text_emb: np.ndarray, reference_motion_emb: np.ndarray, *, bootstrap_replicates: int=BOOTSTRAP_REPLICATES, seed: int=42) -> Dict[str, Any]:
    bundle = text_alignment_bundle(motion_emb, text_emb, reference_motion_emb)
    distributional = {}
    for key in ('fid', 'diversity'):
        distributional[key] = summarize_with_bootstrap([bundle[key]], n_replicates=bootstrap_replicates, seed=seed)
    for key in ('r_precision_top1', 'r_precision_top2', 'r_precision_top3'):
        distributional[key] = {'mean': float(bundle[key]), 'sem': 0.0}
    distributional['mm_dist'] = {'mean': float(bundle['mm_dist']), 'sem': 0.0}
    return {'raw': bundle, 'summary': distributional}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate generated motions on HumanML3D.')
    parser.add_argument('--config', default='configs/evaluate.json')
    parser.add_argument('--data_root', default='')
    parser.add_argument('--generations_dir', required=True)
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--num_samples', type=int, default=NUM_SAMPLES_PER_CAPTION)
    parser.add_argument('--out_json', default='results/eval/metrics.json')
    parser.add_argument('--motion_embeddings', default='', help='Optional .npy [N,D] generated motion embeddings')
    parser.add_argument('--text_embeddings', default='', help='Optional .npy [N,D] matched text embeddings')
    parser.add_argument('--reference_motion_embeddings', default='', help='Optional .npy [N,D] reference motion embeddings')
    add_run_log_cli_args(parser)
    args = parser.parse_args()

    def _run(logger: RunLogger) -> None:
        cfg_path = resolve_repo_path(args.config)
        cfg = load_json(str(cfg_path)) if cfg_path.is_file() else {}
        data_root = Path(resolve_data_root(args.data_root or cfg.get('data_root')))
        generations_dir = Path(args.generations_dir).expanduser().resolve()
        if not generations_dir.is_dir():
            raise FileNotFoundError(f'--generations_dir not found: {generations_dir}')
        motion_ids = load_split_ids(data_root, str(args.split or cfg.get('split', 'test')))
        fps = float(cfg.get('fps', FPS))
        bio = evaluate_biomechanical_split(data_root=data_root, generations_dir=generations_dir, motion_ids=motion_ids, fps=fps, mass_kg=float(cfg.get('mass_kg', 70.0)), height_thresh_m=float(cfg.get('contact_height_thresh_m', 0.06)), speed_thresh_mps=float(cfg.get('contact_speed_thresh_mps', 1.2)))
        result: Dict[str, Any] = {'biomechanical': bio, 'split': str(args.split), 'num_motions': len(bio['per_caption'])}
        if args.motion_embeddings and args.text_embeddings and args.reference_motion_embeddings:
            motion_emb = np.load(args.motion_embeddings)
            text_emb = np.load(args.text_embeddings)
            ref_emb = np.load(args.reference_motion_embeddings)
            result['text'] = evaluate_text_from_embeddings(motion_emb, text_emb, ref_emb, bootstrap_replicates=int(cfg.get('bootstrap_replicates', BOOTSTRAP_REPLICATES)), seed=int(cfg.get('seed', 42)))
        out = Path(args.out_json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding='utf-8')
        logger.progress(f'wrote metrics: {out}')
    run_logged_main(Path(__file__).stem, args.log_dir, _run, argv=sys.argv, no_run_log=bool(args.no_run_log))
if __name__ == '__main__':
    main()