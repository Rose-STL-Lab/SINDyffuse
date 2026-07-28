from __future__ import annotations

from enum import Enum

from common.paths import default_humanml3d_root


class GuidanceMode(str, Enum):
    NONE = "none"
    SINDY = "sindy"


class DatasetName(str, Enum):
    MINT = "mint"


__all__ = ["DatasetName", "GuidanceMode", "default_humanml3d_root"]
