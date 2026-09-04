"""OpenSim-only stubs — nimblephysics removed. Physics guidance uses OpenSim FK."""
from __future__ import annotations
from typing import Any

__all__ = ['NIMBLE_AVAILABLE', 'clear_cache', 'load_model', 'physics_from_q', 'physics_from_q_batch']

NIMBLE_AVAILABLE = False


def clear_cache() -> None:
    return None


def load_model(*_a: Any, **_k: Any) -> Any:
    raise RuntimeError(
        'nimblephysics has been removed. Use OpenSim / nimble.opensim_ik / nimble.lai_features instead.'
    )


def physics_from_q(*_a: Any, **_k: Any) -> Any:
    raise RuntimeError('Nimble physics guidance removed. Use guidance=opensim (nimble.guidance.OpenSimGuidance).')


def physics_from_q_batch(*_a: Any, **_k: Any) -> Any:
    raise RuntimeError('Nimble physics guidance removed. Use guidance=opensim (nimble.guidance.OpenSimGuidance).')
