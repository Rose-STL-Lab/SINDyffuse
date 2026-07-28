"""Load and inspect MinT OpenSim models."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, List, Tuple

from osim.coord_map import default_mint_model_path, mint_coordinate_names


@lru_cache(maxsize=1)
def load_mint_opensim_model() -> Any:
    import opensim as osim

    path = default_mint_model_path()
    model = osim.Model(str(path))
    model.initSystem()
    return model


def mint_muscle_names_from_model() -> Tuple[str, ...]:
    """Muscle names from loaded OpenSim model (lower body only for Lai model)."""
    model = load_mint_opensim_model()
    force_set = model.getForceSet()
    names: List[str] = []
    for i in range(force_set.getSize()):
        force = force_set.get(i)
        if force.getConcreteClassName().endswith("Muscle"):
            names.append(str(force.getName()))
    return tuple(names)


def model_info() -> dict:
    model_path = default_mint_model_path()
    coords = mint_coordinate_names()
    info = {
        "model_path": str(model_path),
        "num_coordinates": len(coords),
        "coordinate_names": list(coords),
    }
    try:
        lu = mint_muscle_names_from_model()
        info["num_lai_muscles"] = len(lu)
        info["lai_muscle_names"] = list(lu)
    except Exception as exc:
        info["lai_muscle_error"] = str(exc)
    return info
