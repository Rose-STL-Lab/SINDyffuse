from __future__ import annotations

from enum import Enum

from common.paths import default_humanml3d_root


class GuidanceMode(str, Enum):
    NONE = "none"
    SINDY = "sindy"
    OSIM = "osim"


class DatasetName(str, Enum):
    HUMANML3D = "humanml3d"


__all__ = ["DatasetName", "GuidanceMode", "default_humanml3d_root"]

