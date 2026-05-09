from __future__ import annotations

import os
from enum import Enum
from pathlib import Path


class GuidanceMode(str, Enum):
    NONE = "none"
    SINDY = "sindy"
    OSIM = "osim"


class DatasetName(str, Enum):
    HUMANML3D = "humanml3d"


def default_humanml3d_root() -> str:
    base = Path(os.environ.get("BIOMECHAI_ROOT", "/mnt/BiomechAI"))
    return str(base / "datasets" / "HumanML3D")

