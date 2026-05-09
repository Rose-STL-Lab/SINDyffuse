from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_legacy_class():
    legacy = Path("/mnt/BiomechAI/experiments/hml3d_diffusion_per_frame/hml_sild_guidance.py")
    if not legacy.exists():
        raise FileNotFoundError(f"Missing legacy SILD guidance source: {legacy}")
    legacy_root = str(legacy.parent)
    if legacy_root not in sys.path:
        sys.path.insert(0, legacy_root)
    spec = importlib.util.spec_from_file_location("legacy_hml_sild_guidance", str(legacy))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load legacy SILD guidance module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LearnedSILDGuidance


LearnedSINDyGuidance = _load_legacy_class()

