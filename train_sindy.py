from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path


def _argv_has_flag(argv: list[str], flag: str) -> bool:
    sep = flag + "="
    return any(a == flag or a.startswith(sep) for a in argv)


def _prepare_argv_for_legacy(legacy_prog_name: str) -> None:
    """Legacy argparse requires ``--output``; Kubeflow/job scripts often omit it.

    Resolution order:

    - If the user passes ``--output`` / ``--data_root``, keep them.
    - Else ``--output``: ``$SINDY_TEXT_OUTPUT_DIR`` or
      ``<SINDyffuse>/results/sindy_text_auto_<timestamp>/``.
    - Else ``--data_root``: ``$HUMANML3D_ROOT`` when set.
    """

    repo_root = Path(__file__).resolve().parent
    user_args = sys.argv[1:]
    merged: list[str] = list(user_args)

    if not _argv_has_flag(merged, "--output"):
        out = os.environ.get("SINDY_TEXT_OUTPUT_DIR", "").strip()
        if out:
            out_dir = Path(out).expanduser()
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = repo_root / "results" / f"sindy_text_auto_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        merged.extend(["--output", str(out_dir)])

    if not _argv_has_flag(merged, "--data_root"):
        data_root = os.environ.get("HUMANML3D_ROOT", "").strip()
        if data_root:
            merged = ["--data_root", data_root] + merged

    sys.argv = [legacy_prog_name] + merged


def main() -> None:
    legacy = Path("/mnt/BiomechAI/experiments/hml3d_diffusion_per_frame/train_hml3d_sild_text.py")
    if not legacy.exists():
        raise FileNotFoundError(f"Missing legacy trainer: {legacy}")
    _prepare_argv_for_legacy(legacy.name)

    legacy_root = str(legacy.parent)
    if legacy_root not in sys.path:
        sys.path.insert(0, legacy_root)

    spec = importlib.util.spec_from_file_location("legacy_train_hml3d_sild_text", str(legacy))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load legacy train_hml3d_sild_text module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    main()

