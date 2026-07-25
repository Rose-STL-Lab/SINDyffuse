"""Canonical MinT muscle ordering (402 strands, MUSINT_402)."""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

MINT_MUSCLE_COUNT = 402


@lru_cache(maxsize=1)
def mint_muscle_names() -> Tuple[str, ...]:
    """Ordered MinT muscle names (80 LU lower body + 322 TL thoracolumbar)."""
    try:
        from musint.benchmarks.muscle_sets import MUSCLE_SUBSETS
    except ImportError as exc:
        raise ImportError(
            "MinT pipeline requires the musint package. "
            "Install with: pip install musint"
        ) from exc
    names = tuple(MUSCLE_SUBSETS["MUSINT_402"])
    if len(names) != MINT_MUSCLE_COUNT:
        raise RuntimeError(
            f"Expected {MINT_MUSCLE_COUNT} MinT muscles, got {len(names)} from musint"
        )
    return names


def validate_activation_matrix(arr, *, atol_zero: float = 1e-8) -> bool:
    """True when activations are all-zero placeholders."""
    import numpy as np

    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return True
    return bool(np.allclose(a, 0.0, atol=atol_zero))
