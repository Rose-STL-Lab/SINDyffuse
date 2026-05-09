from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    legacy = Path("/mnt/BiomechAI/experiments/hml3d_diffusion_per_frame/train_hml3d_sild_text.py")
    if not legacy.exists():
        raise FileNotFoundError(f"Missing legacy trainer: {legacy}")
    spec = importlib.util.spec_from_file_location("legacy_train_hml3d_sild_text", str(legacy))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load legacy train_hml3d_sild_text module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    main()

