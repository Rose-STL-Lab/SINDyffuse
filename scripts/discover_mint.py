#!/usr/bin/env python3
"""Inspect MinT dataset layout, bundled models, and HumanML3D overlap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from osim.discovery import run_discovery, write_discovery_report


def main() -> None:
    parser = argparse.ArgumentParser(description="MinT discovery report")
    parser.add_argument("--mint_root", default="", help="Path to extracted MinT dataset")
    parser.add_argument("--data_root", default="", help="HumanML3D root for overlap stats")
    parser.add_argument("--output", default="", help="Write JSON report to this path")
    args = parser.parse_args()

    kwargs = {}
    if args.mint_root:
        kwargs["mint_root"] = args.mint_root
    if args.data_root:
        kwargs["data_root"] = args.data_root

    report = run_discovery(**kwargs)
    text = json.dumps(report, indent=2)
    if args.output:
        path = write_discovery_report(args.output, **kwargs)
        print(f"Wrote {path}")
    print(text)


if __name__ == "__main__":
    main()
