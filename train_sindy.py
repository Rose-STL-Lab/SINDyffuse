from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path


def _argv_has_flag(argv: list[str], flag: str) -> bool:
    sep = flag + "="
    return any(a == flag or a.startswith(sep) for a in argv)


def _prepare_argv() -> list[str]:
    repo_root = Path(__file__).resolve().parent
    merged = list(sys.argv[1:])

    if not _argv_has_flag(merged, "--output"):
        out = os.environ.get("SINDY_TEXT_OUTPUT_DIR", "").strip()
        out_dir = Path(out).expanduser() if out else repo_root / "results" / f"sindy_{datetime.now():%Y%m%d_%H%M%S}"
        out_dir.mkdir(parents=True, exist_ok=True)
        merged.extend(["--output", str(out_dir)])

    if not _argv_has_flag(merged, "--data_root"):
        data_root = os.environ.get("HUMANML3D_ROOT", "").strip()
        if data_root:
            merged = ["--data_root", data_root] + merged

    return merged


def main() -> None:
    sys.argv = [Path(__file__).name] + _prepare_argv()
    from sindy.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
