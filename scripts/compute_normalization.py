#!/usr/bin/env python3
"""Compute MinT q-space Mean.npy / Std.npy after distributed preprocess."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common.paths import default_humanml3d_root
from common.run_logging import RunLogger, add_run_log_cli_args, null_logger, run_log_session
from datasets.mint_cache_stats import compute_mint_normalization_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute q-space Mean.npy / Std.npy for MinT NPZ cache"
    )
    default_root = default_humanml3d_root()
    parser.add_argument(
        "--out_root",
        default=default_root,
        help="Dataset root containing mint_cache/",
    )
    add_run_log_cli_args(parser)
    args = parser.parse_args()

    out_root = Path(str(args.out_root)).expanduser().resolve()

    def _run(logger: RunLogger) -> None:
        stats = compute_mint_normalization_stats(out_root, split="train")
        logger.progress(json.dumps(stats, indent=2))

    if args.no_run_log:
        _run(null_logger())
        return

    script_name = Path(__file__).stem
    with run_log_session(args.log_dir, script_name=script_name, argv=sys.argv) as (
        paths,
        logger,
    ):
        logger.progress(f"log: {paths.latest_log}")
        _run(logger)


if __name__ == "__main__":
    main()
