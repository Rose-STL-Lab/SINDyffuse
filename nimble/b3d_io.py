"""Legacy B3D I/O — nimblephysics removed. Prefer datasets.lai_cache NPZ."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import warnings

_WARNED = False


def _warn_once() -> None:
    global _WARNED
    if not _WARNED:
        warnings.warn('nimble.b3d_io is obsolete; use datasets.lai_cache', DeprecationWarning, stacklevel=3)
        _WARNED = True


def b3d_file_has_content(path: str | Path, *, min_size: int = 1) -> bool:
    p = Path(path)
    return p.is_file() and p.stat().st_size >= int(min_size)


def try_open_subject_on_disk(path: str | Path) -> Any | None:
    _warn_once()
    del path
    return None


def subject_has_custom_value(subj: Any, name: str) -> bool:
    del subj, name
    return False
