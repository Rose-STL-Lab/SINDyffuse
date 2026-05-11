"""HumanML3D joint indices → Rajagopal2015 marker names (Moco / marker tracking)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import opensim as osim  # type: ignore[import-untyped]

# Each pair: HumanML3D joint index → marker name on ``Rajagopal2015.osim``.
DEFAULT_HML_MARKER_PAIRS: List[Tuple[int, str]] = [
    (1, "LHJC"),
    (2, "RHJC"),
    (4, "LKJC"),
    (5, "RKJC"),
    (7, "LAJC"),
    (8, "RAJC"),
    (10, "LTOE"),
    (11, "RTOE"),
    (12, "C7"),
    (3, "CLAV"),
    (13, "LSJC"),
    (14, "RSJC"),
    (16, "LEJC"),
    (17, "REJC"),
    (18, "LFAradius"),
    (19, "RFAradius"),
]


def marker_names_missing(model_path: str | Path, marker_names: Sequence[str]) -> List[str]:
    """Return marker labels not present on the model (empty if all exist)."""
    mp = Path(model_path)
    missing: List[str] = []
    m = osim.Model(str(mp))
    marker_set = m.getMarkerSet()
    for name in marker_names:
        if not marker_set.has(str(name)):
            missing.append(str(name))
    return missing
