"""Write OpenSim TRC marker trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from mint.ssm67_markers import MARKER_NAMES


def write_trc(
    path: str | Path,
    marker_positions: np.ndarray,
    *,
    marker_names: Sequence[str] = MARKER_NAMES,
    fps: float = 20.0,
) -> None:
    """Write marker trajectories ``[T, M, 3]`` to OpenSim ``.trc`` format."""
    pos = np.asarray(marker_positions, dtype=np.float64)
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError(f"Expected positions [T, M, 3], got {pos.shape}")
    names = list(marker_names)
    if pos.shape[1] != len(names):
        raise ValueError(f"Expected {len(names)} markers, got {pos.shape[1]}")
    t_len = int(pos.shape[0])
    dt = 1.0 / max(float(fps), 1e-8)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "PathFileType  4\t(X/Y/Z)\t" + str(out.resolve()),
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{fps:g}\t{fps:g}\t{t_len}\t{len(names)}\tm\t{fps:g}\t1\t{t_len}",
        "Frame#\tTime\t" + "\t".join(f"{n}\t\t" for n in names),
    ]
    rows = []
    for t in range(t_len):
        row = [str(t + 1), f"{t * dt:.6f}"]
        for m in range(len(names)):
            x, y, z = pos[t, m]
            row.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
        rows.append("\t".join(row))
    out.write_text("\n".join(header + rows) + "\n", encoding="utf-8")
