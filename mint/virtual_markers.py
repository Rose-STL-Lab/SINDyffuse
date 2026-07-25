"""Extract SSM67 virtual markers from SMPL-H mesh vertices."""

from __future__ import annotations

import numpy as np

from mint.ssm67_markers import MARKER_NAMES, VERTEX_INDICES, marker_vertex_map


def extract_ssm67_markers(vertices: np.ndarray) -> np.ndarray:
    """Sample SSM67 markers from SMPL-H mesh ``vertices`` ``[T, V, 3]`` → ``[T, 67, 3]``."""
    v = np.asarray(vertices, dtype=np.float32)
    if v.ndim != 3:
        raise ValueError(f"Expected vertices [T, V, 3], got {v.shape}")
    idx = np.asarray(VERTEX_INDICES, dtype=np.int64)
    if int(idx.max()) >= v.shape[1]:
        raise ValueError(
            f"Vertex index {int(idx.max())} out of range for mesh with {v.shape[1]} vertices"
        )
    markers = v[:, idx, :]
    return markers.astype(np.float32)


def markers_by_name(markers: np.ndarray) -> dict[str, np.ndarray]:
    """``markers`` ``[T, 67, 3]`` → dict name → ``[T, 3]``."""
    if markers.shape[1] != len(MARKER_NAMES):
        raise ValueError(f"Expected [T, {len(MARKER_NAMES)}, 3], got {markers.shape}")
    return {name: markers[:, i, :] for i, name in enumerate(MARKER_NAMES)}


def validate_marker_map() -> None:
    m = marker_vertex_map()
    if len(m) != 67:
        raise ValueError(f"Expected 67 markers, got {len(m)}")
