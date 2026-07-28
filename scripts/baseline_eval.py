#!/usr/bin/env python3
"""Evaluate SINDyffuse and baseline methods with HumanML3D metrics."""

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
from eval.config import BaselineEvalConfig, EvalConfig, MethodSpec
from eval.motion_store import method_motion_lookup
from eval.protocol import compare_methods


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated motions with HumanML3D metrics.")
    parser.add_argument(
        "--config",
        default="configs/baseline_eval.json",
        help="JSON config with methods and eval settings (pass empty string for CLI-only).",
    )
    parser.add_argument("--method", action="append", default=[], help="name=path/to/motions (repeatable)")
    parser.add_argument("--motion_dir", default="", help="Single method motion directory.")
    parser.add_argument("--method_name", default="SINDyffuse")
    parser.add_argument("--data_root", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--evaluator_root", default="")
    parser.add_argument("--output_json", default="results/eval/metrics.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--diversity_times", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    add_run_log_cli_args(parser)
    args = parser.parse_args()

    def _run(logger: RunLogger) -> None:
        if args.config:
            cfg = BaselineEvalConfig.from_json(resolve_repo_path(args.config))
            eval_cfg = cfg.eval
            methods = OrderedDict(
                (m.name, method_motion_lookup(resolve_repo_path(m.motion_dir)))
                for m in cfg.methods
            )
        else:
            eval_cfg = EvalConfig(
                data_root=args.data_root,
                split=args.split,
                batch_size=args.batch_size,
                diversity_times=args.diversity_times,
                evaluator_root=args.evaluator_root,
                device=args.device,
                seed=args.seed,
            ).resolve()
            methods = OrderedDict()
            if args.motion_dir:
                methods[args.method_name] = method_motion_lookup(resolve_repo_path(args.motion_dir))
            for spec in args.method:
                name, path = spec.split("=", 1)
                methods[name.strip()] = method_motion_lookup(resolve_repo_path(path.strip()))

        if not methods:
            raise ValueError("Provide --config, --motion_dir, or --method name=path")

        results = compare_methods(eval_cfg, methods)
        payload = {name: res.to_dict() for name, res in results.items()}
        out = resolve_repo_path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_json(str(out), payload)
        logger.progress(f"saved metrics -> {out}")
        for name, metrics in payload.items():
            logger.progress(
                f"{name}: FID={metrics['fid']:.4f} "
                f"R@1={metrics['r_precision_top1']:.4f} "
                f"MM-Dist={metrics['matching_score']:.4f} "
                f"Diversity={metrics['diversity']:.4f}"
            )

    run_logged_main(Path(__file__).stem, args.log_dir, _run, argv=sys.argv, no_run_log=bool(args.no_run_log))


if __name__ == "__main__":
    main()
