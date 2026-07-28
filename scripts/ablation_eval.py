#!/usr/bin/env python3
"""Guidance ablation: generate variants, then HumanML3D + biomechanical metrics."""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.io import save_json
from common.paths import resolve_repo_path
from common.run_logging import RunLogger, add_run_log_cli_args, run_logged_main
from eval.biomech_metrics import evaluate_biomech_from_generated
from eval.config import AblationEvalConfig
from eval.generate import GenerateConfig, generate_batch
from eval.motion_store import method_motion_lookup
from eval.protocol import compare_methods


def main() -> None:
    parser = argparse.ArgumentParser(description="Guidance ablation evaluation.")
    parser.add_argument(
        "--config",
        default="configs/ablation_eval.json",
        help="Ablation JSON config.",
    )
    parser.add_argument("--skip_generate", action="store_true")
    parser.add_argument("--skip_humanml", action="store_true")
    parser.add_argument("--skip_biomech", action="store_true")
    parser.add_argument("--output_json", default="")
    add_run_log_cli_args(parser)
    args = parser.parse_args()

    def _run(logger: RunLogger) -> None:
        cfg = AblationEvalConfig.from_json(resolve_repo_path(args.config))
        output_root = resolve_repo_path(cfg.output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        variant_dirs: OrderedDict[str, Path] = OrderedDict()
        for variant in cfg.variants:
            variant_dir = output_root / variant.name
            variant_dirs[variant.name] = variant_dir
            if args.skip_generate:
                logger.progress(f"skip generate: {variant.name}")
                continue

            checkpoint = variant.checkpoint or cfg.diffusion_checkpoint
            if not checkpoint:
                raise ValueError(f"No checkpoint for variant {variant.name}")

            gen_cfg = GenerateConfig(
                checkpoint=checkpoint,
                data_root=cfg.eval.data_root,
                split=cfg.eval.split,
                guidance=variant.guidance,
                sindy_checkpoint_dir=cfg.sindy_checkpoint_dir,
                surrogate_checkpoint_dir=cfg.surrogate_checkpoint_dir,
                output_dir=str(variant_dir),
                variant=variant.name,
                cfg_scale=variant.cfg_scale,
                opt_steps=variant.opt_steps,
                opt_lr=variant.opt_lr,
                num_samples_per_caption=cfg.num_samples_per_caption,
                max_motions=cfg.max_motions,
                device=cfg.eval.device,
                seed=cfg.eval.seed,
            )
            records = generate_batch(gen_cfg)
            logger.progress(f"{variant.name}: generated {len(records)} motions")

        results: dict[str, dict] = {}

        if not args.skip_humanml:
            methods = OrderedDict(
                (name, method_motion_lookup(path)) for name, path in variant_dirs.items()
            )
            humanml = compare_methods(cfg.eval, methods)
            for name, metrics in humanml.items():
                results.setdefault(name, {})["humanml3d"] = metrics.to_dict()

        if not args.skip_biomech:
            for name, path in variant_dirs.items():
                try:
                    bio = evaluate_biomech_from_generated(path, data_root=cfg.eval.data_root)
                    results.setdefault(name, {})["biomech"] = bio.to_dict()
                except ValueError as exc:
                    logger.progress(f"{name}: biomech skipped ({exc})")

        out = resolve_repo_path(args.output_json or str(output_root / "ablation_results.json"))
        save_json(str(out), results)
        logger.progress(f"saved ablation results -> {out}")

    run_logged_main(Path(__file__).stem, args.log_dir, _run, argv=sys.argv, no_run_log=bool(args.no_run_log))


if __name__ == "__main__":
    main()
