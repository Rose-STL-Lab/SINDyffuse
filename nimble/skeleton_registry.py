"""Skeleton registry stub — nimblephysics Rajagopal model removed."""
from __future__ import annotations
from typing import Any


def get_spec(name: str = 'rajagopal') -> Any:
    raise RuntimeError(f'nimblephysics skeleton {name!r} removed; use LaiUhlrich2022 via OpenSim')


def load_skeleton(name: str = 'rajagopal', *, with_geometry: bool = True) -> Any:
    del with_geometry
    raise RuntimeError(f'nimblephysics skeleton {name!r} removed; use LaiUhlrich2022 via OpenSim')


SKELETONS: dict = {}
