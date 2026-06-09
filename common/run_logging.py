"""Dual-channel run logging: minimal terminal progress, verbose details in ``logs/``."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional, TextIO, TypeVar

_T = TypeVar("_T")


class _TeeStream:
    """Write-through wrapper duplicating stderr to a log file."""

    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._primary.write(data)
        self._secondary.write(data)
        self._primary.flush()
        self._secondary.flush()
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def fileno(self) -> int:
        return self._primary.fileno()

    def isatty(self) -> bool:
        return getattr(self._primary, "isatty", lambda: False)()


@dataclass(frozen=True)
class RunLogPaths:
    log_dir: Path
    log_file: Path
    latest_log: Path
    latest_exit_code: Path


class RunLogger:
    """Terminal progress vs file-only verbose logging."""

    def __init__(
        self,
        *,
        terminal: TextIO,
        log_file: TextIO | None,
        progress_enabled: bool = True,
    ) -> None:
        self._terminal = terminal
        self._log_file = log_file
        self._progress_enabled = progress_enabled

    @property
    def enabled(self) -> bool:
        return self._log_file is not None

    def progress(self, msg: str) -> None:
        if self._progress_enabled:
            print(msg, file=self._terminal, flush=True)

    def verbose(self, msg: str) -> None:
        if self._log_file is not None:
            print(msg, file=self._log_file, flush=True)
        elif self._progress_enabled:
            print(msg, file=self._terminal, flush=True)

    def warn(self, msg: str) -> None:
        print(msg, file=self._terminal, flush=True)
        if self._log_file is not None:
            print(msg, file=self._log_file, flush=True)

    def log_exception(self, msg: str, *, exc: BaseException | None = None) -> None:
        self.progress(msg)
        if self._log_file is not None:
            print(msg, file=self._log_file, flush=True)
            if exc is not None:
                traceback.print_exception(type(exc), exc, exc.__traceback__, file=self._log_file)
            else:
                traceback.print_exc(file=self._log_file)


_NULL_LOGGER: RunLogger | None = None


def null_logger() -> RunLogger:
    """Logger that mirrors everything to stdout (``--no_run_log``)."""
    global _NULL_LOGGER
    if _NULL_LOGGER is None:
        _NULL_LOGGER = RunLogger(terminal=sys.stdout, log_file=None)
    return _NULL_LOGGER


_ACTIVE_LOGGER: RunLogger | None = None


def get_run_logger() -> RunLogger:
    return _ACTIVE_LOGGER if _ACTIVE_LOGGER is not None else null_logger()


def append_verbose_log(msg: str) -> None:
    """Write a line to the active run log or ``SINDYFFUSE_VERBOSE_LOG`` (worker processes)."""
    active = _ACTIVE_LOGGER
    if active is not None and active.enabled:
        active.verbose(msg)
        return
    path = os.environ.get("SINDYFFUSE_VERBOSE_LOG", "").strip()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fp:
            print(msg, file=fp, flush=True)
    except OSError:
        pass


def set_active_logger(logger: RunLogger | None) -> None:
    global _ACTIVE_LOGGER
    _ACTIVE_LOGGER = logger


def default_log_dir(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parent.parent
    return root / "logs"


def build_run_log_paths(
    log_dir: str | Path,
    *,
    script_name: str,
    rank: int | None = None,
) -> RunLogPaths:
    """Return paths for ``{script_name}_{timestamp}.log`` and ``{script_name}_latest``."""
    directory = Path(log_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = Path(script_name).stem.strip().replace("/", "_") or "run"
    if rank is not None and rank > 0:
        safe_name = f"{safe_name}_rank{rank}"
    log_file = directory / f"{safe_name}_{stamp}.log"
    latest_stem = safe_name if rank is None or rank == 0 else f"{Path(script_name).stem}_latest"
    return RunLogPaths(
        log_dir=directory,
        log_file=log_file,
        latest_log=directory / f"{latest_stem}.log",
        latest_exit_code=directory / f"{Path(script_name).stem}_latest.exitcode",
    )


def add_run_log_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log_dir",
        type=str,
        default=str(default_log_dir()),
        help="Directory for timestamped run logs (created if missing).",
    )
    parser.add_argument(
        "--no_run_log",
        action="store_true",
        help="Disable verbose file logging under logs/.",
    )


def _update_latest_link(target: Path, latest: Path) -> None:
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(target.name)
    except OSError:
        latest.write_bytes(target.read_bytes())


@contextmanager
def run_log_session(
    log_dir: str | Path,
    *,
    script_name: str,
    argv: Optional[list[str]] = None,
    progress_enabled: bool = True,
    rank: int | None = None,
) -> Iterator[tuple[RunLogPaths, RunLogger]]:
    """Mirror stderr to log file; use ``RunLogger`` for stdout channels."""
    paths = build_run_log_paths(log_dir, script_name=script_name, rank=rank)
    log_fp = paths.log_file.open("a", encoding="utf-8", buffering=1)
    prev_stderr = sys.stderr
    logger = RunLogger(
        terminal=sys.stdout,
        log_file=log_fp,
        progress_enabled=progress_enabled,
    )
    sys.stderr = _TeeStream(prev_stderr, log_fp)  # type: ignore[assignment]

    logger.verbose(f"=== run log start {datetime.now(timezone.utc).isoformat()} ===")
    logger.verbose(f"log_file={paths.log_file}")
    if argv is not None:
        logger.verbose(f"argv={' '.join(argv)}")

    exit_code = 0
    set_active_logger(logger)
    try:
        yield paths, logger
    except SystemExit as exc:
        if isinstance(exc.code, int):
            exit_code = exc.code
        elif exc.code is None:
            exit_code = 0
        else:
            exit_code = 1
        raise
    except BaseException as exc:
        exit_code = 1
        logger.log_exception("Unhandled exception", exc=exc)
        raise
    finally:
        set_active_logger(None)
        sys.stderr = prev_stderr
        try:
            print(f"=== run log end exit_code={exit_code} ===", file=log_fp, flush=True)
        finally:
            log_fp.close()
        if rank is None or rank == 0:
            _update_latest_link(paths.log_file, paths.latest_log)
            paths.latest_exit_code.write_text(f"{exit_code}\n", encoding="utf-8")


def run_logged_main(
    script_name: str,
    log_dir: str | Path,
    fn: Callable[[RunLogger], _T],
    *,
    argv: Optional[list[str]] = None,
    no_run_log: bool = False,
    progress_enabled: bool = True,
    rank: int | None = None,
) -> _T:
    """Run ``fn(logger)`` inside a file log session unless ``no_run_log``."""
    if no_run_log:
        return fn(null_logger())
    with run_log_session(
        log_dir,
        script_name=script_name,
        argv=argv,
        progress_enabled=progress_enabled,
        rank=rank,
    ) as (paths, logger):
        logger.progress(f"log: {paths.latest_log}")
        return fn(logger)
