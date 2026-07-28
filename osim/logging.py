"""OpenSim logger configuration helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager


def _normalize_opensim_log_level(level: str) -> str:
    key = str(level).strip().lower()
    if key in ("", "silent", "none", "quiet"):
        return "Off"
    return str(level).strip()


def configure_opensim_logging(level: str = "Off") -> None:
    """Set global OpenSim logger level (call early in worker / before loading models)."""
    try:
        import opensim as osim

        osim.Logger.setLevelString(_normalize_opensim_log_level(level))
    except Exception:
        pass


def _opensim_stdio_suppressed(level: str) -> bool:
    return _normalize_opensim_log_level(level) == "Off"


@contextmanager
def _suppress_process_stdio():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        os.close(devnull_fd)


@contextmanager
def opensim_quiet(level: str = "Off"):
    """Suppress OpenSim Logger + console during solves."""
    import opensim as osim

    normalized = _normalize_opensim_log_level(level)
    prev = osim.Logger.getLevelString()
    configure_opensim_logging(normalized)
    if _opensim_stdio_suppressed(normalized):
        with _suppress_process_stdio():
            try:
                yield
            finally:
                try:
                    osim.Logger.setLevelString(prev)
                except Exception:
                    pass
    else:
        try:
            yield
        finally:
            try:
                osim.Logger.setLevelString(prev)
            except Exception:
                pass


__all__ = ["configure_opensim_logging", "opensim_quiet"]
