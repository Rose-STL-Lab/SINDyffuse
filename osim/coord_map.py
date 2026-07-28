"""OpenSim coordinate layout for the MinT Lai model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from common.paths import repo_root

# Populated on first model load from OpenSim.
_FALLBACK_NDOF = 33


@lru_cache(maxsize=1)
def default_mint_model_path() -> Path:
    env = __import__("os").environ.get("MINT_OSIM_PATH", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
    bundled = repo_root() / "osim" / "models" / "LaiUhlrich2022.osim"
    if bundled.is_file():
        return bundled
    raise FileNotFoundError(
        "MinT OpenSim model not found. Set MINT_OSIM_PATH or place "
        "LaiUhlrich2022.osim under osim/models/ (see osim/models/README.md)."
    )


@lru_cache(maxsize=1)
def mint_coordinate_names() -> Tuple[str, ...]:
    try:
        import opensim as osim

        model = osim.Model(str(default_mint_model_path()))
        model.initSystem()
        names: List[str] = []
        coord_set = model.getCoordinateSet()
        for i in range(coord_set.getSize()):
            names.append(str(coord_set.get(i).getName()))
        return tuple(names)
    except Exception:
        return tuple(f"mint_coord_{i}" for i in range(_FALLBACK_NDOF))


def mint_coordinate_count() -> int:
    return len(mint_coordinate_names())


def q_layout_description() -> str:
    return (
        f"MinT q vector [{mint_coordinate_count()}]: "
        + ", ".join(mint_coordinate_names()[:8])
        + ("..." if mint_coordinate_count() > 8 else "")
    )
