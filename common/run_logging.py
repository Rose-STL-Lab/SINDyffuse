"""Run logging to terminal and optional timestamped log files under ``logs/``."""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional, TextIO, TypeVar

from tqdm import tqdm

_T = TypeVar("_T")


class _LockedAppendFile:
    """Append-only log handle safe for concurrent writers on a shared filesystem."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fp = path.open("a", encoding="utf-8", buffering=1)

    def write(self, data: str) -> int:
        if not data:
            return 0
        fcntl.flock(self._fp.fileno(), fcntl.LOCK_EX)
        try:
            self._fp.write(data)
            self._fp.flush()
        finally:
            fcntl.flock(self._fp.fileno(), fcntl.LOCK_UN)
        return len(data)

    def flush(self) -> None:
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()

    def fileno(self) -> int:
        return self._fp.fileno()


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
    """Mirror progress and verbose messages to terminal and an optional log file."""

    def __init__(
        self,
        *,
        terminal: TextIO,
        log_file: TextIO | None,
        progress_enabled: bool = True,
        shared_log: bool = False,
        line_prefix: str = "",
    ) -> None:
        self._terminal = terminal
        self._log_file = log_file
        self._progress_enabled = progress_enabled
        self._shared_log = shared_log
        self._line_prefix = line_prefix

    @property
    def enabled(self) -> bool:
        return self._log_file is not None

    @property
    def terminal(self) -> TextIO:
        return self._terminal

    @property
    def log_file_stream(self) -> TextIO | None:
        return self._log_file

    @property
    def shared_log(self) -> bool:
        return self._shared_log

    def _format(self, msg: str) -> str:
        if self._line_prefix and msg:
            return f"{self._line_prefix}{msg}"
        return msg

    def _emit(self, msg: str) -> None:
        terminal_msg = msg
        file_msg = self._format(msg)
        if self._progress_enabled:
            print(terminal_msg, file=self._terminal, flush=True)
        if self._log_file is not None:
            print(file_msg, file=self._log_file, flush=True)

    def progress(self, msg: str) -> None:
        self._emit(msg)

    def verbose(self, msg: str) -> None:
        self._emit(msg)

    def warn(self, msg: str) -> None:
        self._emit(msg)

    def log_exception(self, msg: str, *, exc: BaseException | None = None) -> None:
        self._emit(msg)
        if exc is not None:
            if self._progress_enabled:
                traceback.print_exception(type(exc), exc, exc.__traceback__)
            if self._log_file is not None:
                traceback.print_exception(
                    type(exc), exc, exc.__traceback__, file=self._log_file
                )
        else:
            if self._progress_enabled:
                traceback.print_exc()
            if self._log_file is not None:
                traceback.print_exc(file=self._log_file)


class DualTqdm:
    """Progress bar mirrored to terminal and an optional log file stream."""

    def __init__(
        self,
        *,
        total: int,
        desc: str = "",
        unit: str = "",
        terminal: TextIO,
        log_file: TextIO | None = None,
    ) -> None:
        self._bars: list[tqdm] = [
            tqdm(
                total=total,
                desc=desc,
                unit=unit,
                file=terminal,
                dynamic_ncols=True,
            )
        ]
        if log_file is not None:
            self._bars.append(
                tqdm(
                    total=total,
                    desc=desc,
                    unit=unit,
                    file=log_file,
                    ascii=True,
                    dynamic_ncols=False,
                    mininterval=1.0,
                )
            )

    def update(self, n: int = 1) -> None:
        for bar in self._bars:
            bar.update(n)

    def set_postfix(self, ordered_dict=None, refresh: bool = True, **kwargs) -> None:
        for bar in self._bars:
            bar.set_postfix(ordered_dict=ordered_dict, refresh=refresh, **kwargs)

    def close(self) -> None:
        for bar in self._bars:
            bar.close()

    def __enter__(self) -> DualTqdm:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def dual_tqdm(
    *,
    total: int,
    desc: str = "",
    unit: str = "",
    logger: RunLogger | None = None,
) -> DualTqdm:
    """Create a progress bar on the active logger's terminal and log file."""
    log = logger if logger is not None else get_run_logger()
    log_file = None if log.shared_log else log.log_file_stream
    return DualTqdm(
        total=total,
        desc=desc,
        unit=unit,
        terminal=log.terminal,
        log_file=log_file,
    )


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
    prefix = os.environ.get("SINDYFFUSE_VERBOSE_LOG_PREFIX", "").strip()
    if prefix:
        msg = f"{prefix} {msg}" if not prefix.endswith(" ") else f"{prefix}{msg}"
    active = _ACTIVE_LOGGER
    if active is not None and active.enabled:
        active.verbose(msg)
        return
    path = os.environ.get("SINDYFFUSE_VERBOSE_LOG", "").strip()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fp:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
            try:
                print(msg, file=fp, flush=True)
            finally:
                fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def set_active_logger(logger: RunLogger | None) -> None:
    global _ACTIVE_LOGGER
    _ACTIVE_LOGGER = logger


def default_log_dir(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parent.parent
    return root / "logs"


def resolve_run_log_id(explicit: str | None = None) -> str | None:
    """Shared run id for distributed jobs (``SINDYFFUSE_RUN_LOG_ID`` env or CLI)."""
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    env = os.environ.get("SINDYFFUSE_RUN_LOG_ID", "").strip()
    return env or None


def build_run_log_paths(
    log_dir: str | Path,
    *,
    script_name: str,
    rank: int | None = None,
    run_id: str | None = None,
) -> RunLogPaths:
    """Return paths for per-run logs and the ``{script_name}.log`` latest symlink."""
    if isinstance(log_dir, argparse.Namespace):
        raise TypeError(
            "log_dir must be a path string, not argparse.Namespace; pass args.log_dir"
        )
    directory = Path(log_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = Path(script_name).stem.strip().replace("/", "_") or "run"
    if rank is not None and rank > 0:
        safe_name = f"{safe_name}_rank{rank}"

    resolved_run_id = resolve_run_log_id(run_id)
    if resolved_run_id:
        log_file = directory / f"{safe_name}_{resolved_run_id}.log"
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_file = directory / f"{safe_name}_{stamp}.log"

    latest_stem = safe_name if rank is None or rank == 0 else Path(script_name).stem
    return RunLogPaths(
        log_dir=directory,
        log_file=log_file,
        latest_log=directory / f"{latest_stem}.log",
        latest_exit_code=directory / f"{Path(script_name).stem}.exitcode",
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
    parser.add_argument(
        "--run_log_id",
        type=str,
        default="",
        help="Shared run id for distributed shards (default: SINDYFFUSE_RUN_LOG_ID env).",
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
    run_id: str | None = None,
    line_prefix: str = "",
) -> Iterator[tuple[RunLogPaths, RunLogger]]:
    """Mirror stderr to log file; use ``RunLogger`` for stdout channels."""
    resolved_run_id = resolve_run_log_id(run_id)
    paths = build_run_log_paths(
        log_dir,
        script_name=script_name,
        rank=rank,
        run_id=resolved_run_id,
    )
    shared_log = resolved_run_id is not None
    log_fp: TextIO
    if shared_log:
        log_fp = _LockedAppendFile(paths.log_file)  # type: ignore[assignment]
    else:
        log_fp = paths.log_file.open("a", encoding="utf-8", buffering=1)
    prev_stderr = sys.stderr
    logger = RunLogger(
        terminal=sys.stdout,
        log_file=log_fp,
        progress_enabled=progress_enabled,
        shared_log=shared_log,
        line_prefix=line_prefix,
    )
    sys.stderr = _TeeStream(prev_stderr, log_fp)  # type: ignore[assignment]

    logger.verbose(f"=== run log start {datetime.now(timezone.utc).isoformat()} ===")
    logger.verbose(f"log_file={paths.log_file}")
    if resolved_run_id:
        logger.verbose(f"run_log_id={resolved_run_id}")
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
        if resolved_run_id:
            _update_latest_link(paths.log_file, paths.latest_log)
        elif rank is None or rank == 0:
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
    run_id: str | None = None,
    line_prefix: str = "",
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
        run_id=run_id,
        line_prefix=line_prefix,
    ) as (paths, logger):
        logger.progress(f"log: {paths.latest_log}")
        return fn(logger)
