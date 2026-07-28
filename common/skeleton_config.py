"""Skeleton dimension constants for the MinT OpenSim pipeline."""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

from common.biomech import BIOMECH_COMPONENT_KEYS
from common.paths import MINT_CACHE_SUBDIR

DEFAULT_FPS = 20.0
MINT_MUSCLE_COUNT = 402


def cache_subdir() -> str:
    return MINT_CACHE_SUBDIR


def muscle_activation_rows() -> int:
    return int(MINT_MUSCLE_COUNT)


@lru_cache(maxsize=1)
def mint_ndof() -> int:
    from osim.coord_map import mint_coordinate_count

    return int(mint_coordinate_count())


def motion_ndof() -> int:
    return mint_ndof()


def n_bio_targets() -> int:
    return len(BIOMECH_COMPONENT_KEYS)


def n_muscle_targets() -> int:
    return muscle_activation_rows()


def n_sindy_targets() -> int:
    return n_bio_targets() + n_muscle_targets()


def muscle_channel_names() -> Tuple[str, ...]:
    from osim.muscle_schema import mint_muscle_names

    return mint_muscle_names()


__all__ = [
    "DEFAULT_FPS",
    "MINT_MUSCLE_COUNT",
    "cache_subdir",
    "mint_ndof",
    "motion_ndof",
    "muscle_activation_rows",
    "muscle_channel_names",
    "n_bio_targets",
    "n_muscle_targets",
    "n_sindy_targets",
]
