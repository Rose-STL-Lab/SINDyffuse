from __future__ import annotations
from enum import Enum
from common.paths import default_humanml3d_root

class GuidanceMode(str, Enum):
    NONE = 'none'
    SINDY = 'sindy'
    OPENSIM = 'opensim'
    # Deprecated alias
    NIMBLE = 'opensim'

class DatasetName(str, Enum):
    NIMBLE = 'nimble'  # Lai NPZ dataset (legacy name)
    LAI = 'lai'
__all__ = ['DatasetName', 'GuidanceMode', 'default_humanml3d_root']
