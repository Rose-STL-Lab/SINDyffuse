"""HumanML3D evaluation protocol."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from common.runtime import resolve_torch_device, set_seed
from eval.config import EvalConfig
from eval.dataset import HumanML3DEvalDataset, build_eval_dataloader
from eval.evaluator_wrapper import EvaluatorConfig, EvaluatorModelWrapper
from eval.metrics import (
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_top_k,
    euclidean_distance_matrix,
)


@dataclass
class EvalResults:
    matching_score: float
    r_precision_top1: float
    r_precision_top2: float
    r_precision_top3: float
    fid: float
    diversity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _evaluate_loader(
    eval_wrapper: EvaluatorModelWrapper,
    loader,
    *,
    include_text: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    all_motion_embeddings = []
    matching_score_sum = 0.0
    top_k_count = np.zeros(3, dtype=np.float64)
    all_size = 0

    with torch.no_grad():
        for batch in loader:
            word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
            if include_text:
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens,
                )
                dist_mat = euclidean_distance_matrix(
                    text_embeddings.cpu().numpy(),
                    motion_embeddings.cpu().numpy(),
                )
                matching_score_sum += float(dist_mat.trace())
                argsmax = np.argsort(dist_mat, axis=1)
                top_k_count += calculate_top_k(argsmax, top_k=3).sum(axis=0)
            else:
                motion_embeddings = eval_wrapper.get_motion_embeddings(motions, m_lens)

            all_size += motion_embeddings.shape[0]
            all_motion_embeddings.append(motion_embeddings.cpu().numpy())

    motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
    matching_score = matching_score_sum / max(all_size, 1)
    r_precision = top_k_count / max(all_size, 1)
    return matching_score, r_precision, motion_embeddings, top_k_count


def run_humanml3d_eval(
    cfg: EvalConfig,
    motion_lookup: dict[str, np.ndarray] | None = None,
    *,
    method_name: str = "method",
    diversity_times: int | None = None,
) -> EvalResults:
    cfg = cfg.resolve()
    set_seed(cfg.seed)
    device = resolve_torch_device(cfg.device)
    evaluator_cfg = EvaluatorConfig.from_paths(cfg.evaluator_root, device=str(device))
    eval_wrapper = EvaluatorModelWrapper(evaluator_cfg)

    gt_dataset = HumanML3DEvalDataset(cfg, split=cfg.split)
    gt_loader = build_eval_dataloader(gt_dataset, batch_size=cfg.batch_size, shuffle=False)

    if motion_lookup is None:
        gen_dataset = gt_dataset
    else:
        gen_dataset = HumanML3DEvalDataset(cfg, motion_lookup=motion_lookup, split=cfg.split)
    gen_loader = build_eval_dataloader(gen_dataset, batch_size=cfg.batch_size, shuffle=False)

    matching_score, r_precision, gen_embeddings, _ = _evaluate_loader(
        eval_wrapper,
        gen_loader,
        include_text=True,
    )
    _, _, gt_embeddings, _ = _evaluate_loader(
        eval_wrapper,
        gt_loader,
        include_text=False,
    )

    gt_mu, gt_cov = calculate_activation_statistics(gt_embeddings)
    gen_mu, gen_cov = calculate_activation_statistics(gen_embeddings)
    fid = calculate_frechet_distance(gt_mu, gt_cov, gen_mu, gen_cov)
    diversity = calculate_diversity(gen_embeddings, diversity_times or cfg.diversity_times)

    results = EvalResults(
        matching_score=float(matching_score),
        r_precision_top1=float(r_precision[0]),
        r_precision_top2=float(r_precision[1]),
        r_precision_top3=float(r_precision[2]),
        fid=float(fid),
        diversity=float(diversity),
    )
    print(f"[{method_name}] MM-Dist={results.matching_score:.4f} "
          f"R@1={results.r_precision_top1:.4f} R@2={results.r_precision_top2:.4f} "
          f"R@3={results.r_precision_top3:.4f} FID={results.fid:.4f} "
          f"Diversity={results.diversity:.4f}")
    return results


def compare_methods(
    cfg: EvalConfig,
    methods: OrderedDict[str, dict[str, np.ndarray]],
) -> OrderedDict[str, EvalResults]:
    all_results: OrderedDict[str, EvalResults] = OrderedDict()
    for name, motion_lookup in methods.items():
        all_results[name] = run_humanml3d_eval(cfg, motion_lookup, method_name=name)
    return all_results
