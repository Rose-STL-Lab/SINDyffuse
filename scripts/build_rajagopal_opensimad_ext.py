#!/usr/bin/env python3
"""One-shot: prepare AD-ready Rajagopal + OpenSimAD external function F."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# CasADi must load before OpenSim in-process: OpenSim ships libcasadi.so.3.7 in the
# conda env lib/ and loading it first breaks the pip CasADi extension import.
import casadi  # noqa: F401
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

def main() -> None:
    parser = argparse.ArgumentParser(description='Build Rajagopal OpenSimAD external function (MinT/OpenCap toolchain)')
    parser.add_argument('--force', action='store_true', help='Rebuild even if artifacts exist')
    parser.add_argument('--no_expression_graph', action='store_true', help='Build classic .so external instead of expression-graph .py')
    args = parser.parse_args()
    from nimble.opensimad.codegen import build_rajagopal_opensimad_external
    from nimble.opensimad.model_prep import ensure_ad_ready_artifacts
    from nimble.opensimad.paths import external_function_dir
    base, contacts = ensure_ad_ready_artifacts(force=bool(args.force))
    print(f'AD base model: {base}')
    print(f'AD contacts model: {contacts}')
    out = build_rajagopal_opensimad_external(force=bool(args.force), use_expression_graph=not bool(args.no_expression_graph))
    print(f'ExternalFunction dir: {out}')
    for name in sorted(p.name for p in external_function_dir().iterdir()):
        print(f'  {name}')

if __name__ == '__main__':
    main()
