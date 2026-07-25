"""Skeleton selection and dimension constants for Rajagopal vs MinT pipelines."""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from typing import Tuple

from common.paths import MINT_CACHE_SUBDIR, NIMBLE_B3D_SUBDIR


class SkeletonKind(str, Enum):
    RAJAGOPAL = "rajagopal"
    MINT = "mint"


DEFAULT_SKELETON_ENV = "SINDYFFUSE_SKELETON"
DEFAULT_FPS = 20.0

# Rajagopal (bundled Nimble model)
RAJAGOPAL_NDOF = 37
RAJAGOPAL_MUSCLE_COUNT = 80

# MinT (Lai lower body + Bruno thoracolumbar; ndof resolved from OpenSim model)
MINT_MUSCLE_COUNT = 402


def resolve_skeleton(kind: str | SkeletonKind | None = None) -> SkeletonKind:
    if kind is None or str(kind).strip() == "":
        raw = os.environ.get(DEFAULT_SKELETON_ENV, SkeletonKind.MINT.value).strip().lower()
        return SkeletonKind(raw)
    return SkeletonKind(str(kind).strip().lower())


def cache_subdir(kind: SkeletonKind | str | None = None) -> str:
    sk = resolve_skeleton(kind)
    if sk == SkeletonKind.MINT:
        return MINT_CACHE_SUBDIR
    return NIMBLE_B3D_SUBDIR


def muscle_activation_rows(kind: SkeletonKind | str | None = None) -> int:
    sk = resolve_skeleton(kind)
    if sk == SkeletonKind.MINT:
        return int(MINT_MUSCLE_COUNT)
    return int(RAJAGOPAL_MUSCLE_COUNT)


@lru_cache(maxsize=1)
def mint_ndof() -> int:
    from mint.coord_map import mint_coordinate_count

    return int(mint_coordinate_count())


def motion_ndof(kind: SkeletonKind | str | None = None) -> int:
    sk = resolve_skeleton(kind)
    if sk == SkeletonKind.MINT:
        return mint_ndof()
    return int(RAJAGOPAL_NDOF)


def n_bio_targets() -> int:
    from nimble.channels import BIOMECH_COMPONENT_KEYS

    return len(BIOMECH_COMPONENT_KEYS)


def n_muscle_targets(kind: SkeletonKind | str | None = None) -> int:
    return muscle_activation_rows(kind)


def n_sindy_targets(kind: SkeletonKind | str | None = None) -> int:
    return n_bio_targets() + n_muscle_targets(kind)


def muscle_channel_names(kind: SkeletonKind | str | None = None) -> Tuple[str, ...]:
    sk = resolve_skeleton(kind)
    if sk == SkeletonKind.MINT:
        from mint.muscle_schema import mint_muscle_names

        return mint_muscle_names()
    from nimble.muscle_activation import muscle_names, opensim_quiet

    with opensim_quiet("Off"):
        return tuple(muscle_names())


__all__ = [
    "DEFAULT_FPS",
    "DEFAULT_SKELETON_ENV",
    "MINT_MUSCLE_COUNT",
    "RAJAGOPAL_MUSCLE_COUNT",
    "RAJAGOPAL_NDOF",
    "SkeletonKind",
    "cache_subdir",
    "mint_ndof",
    "motion_ndof",
    "muscle_activation_rows",
    "muscle_channel_names",
    "n_bio_targets",
    "n_muscle_targets",
    "n_sindy_targets",
    "resolve_skeleton",
]
