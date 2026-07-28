#!/usr/bin/env python3
"""Generate motions on the HumanML3D test split for evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.io import save_json
from common.run_logging import RunLogger, add_run_log_cli_args, run_logged_main
from eval.generate import GenerateConfig, generate_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-generate motions for evaluation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="results/eval/generated")
    parser.add_argument("--config", default="", help="Optional JSON config override.")
    parser.add_argument("--data_root", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--guidance", choices=["none", "sindy"], default="sindy")
    parser.add_argument("--sindy_checkpoint_dir", default="")
    parser.add_argument("--surrogate_checkpoint_dir", default="")
    parser.add_argument("--variant", default="default")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--cfg_scale", type=float, default=2.5)
    parser.add_argument("--opt_steps", type=int, default=0)
    parser.add_argument("--opt_lr", type=float, default=1e-3)
    parser.add_argument("--num_samples_per_caption", type=int, default=1)
    parser.add_argument("--max_motions", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    add_run_log_cli_args(parser)
    args = parser.parse_args()

    def _run(logger: RunLogger) -> None:
        cfg = GenerateConfig(
            checkpoint=args.checkpoint,
            data_root=args.data_root,
            split=args.split,
            guidance=args.guidance,
            sindy_checkpoint_dir=args.sindy_checkpoint_dir,
            surrogate_checkpoint_dir=args.surrogate_checkpoint_dir,
            output_dir=args.output_dir,
            variant=args.variant,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            timesteps=args.timesteps,
            cfg_scale=args.cfg_scale,
            opt_steps=args.opt_steps,
            opt_lr=args.opt_lr,
            num_samples_per_caption=args.num_samples_per_caption,
            max_motions=args.max_motions,
            device=args.device,
            seed=args.seed,
        )
        if args.config:
            payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
            for key, value in payload.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
        records = generate_batch(cfg)
        manifest = {
            "variant": cfg.variant,
            "guidance": cfg.guidance,
            "num_records": len(records),
            "output_dir": str(cfg.output_dir),
            "records": [
                {
                    "sample_id": r.sample_id,
                    "motion_id": r.motion_id,
                    "caption": r.caption,
                    "length": r.length,
                    "variant": r.variant,
                }
                for r in records
            ],
        }
        out_dir = Path(cfg.output_dir)
        save_json(str(out_dir / "manifest.json"), manifest)
        logger.progress(f"generated {len(records)} motions -> {out_dir}")

    run_logged_main(Path(__file__).stem, args.log_dir, _run, argv=sys.argv, no_run_log=bool(args.no_run_log))


if __name__ == "__main__":
    main()
