"""Dependency-free, process-safe diagnostics that survive disposable workers."""
from __future__ import annotations
import fcntl
import json
import os
import resource
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

DIAGNOSTICS_ENV = 'SINDYFFUSE_POLYNOMIAL_DIAGNOSTICS'

def memory_snapshot() -> dict:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {'rss_peak_bytes': int(peak if sys.platform == 'darwin' else peak * 1024)}
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                result['rss_current_bytes'] = int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    # Containers normally expose their own cgroup at these mount roots.
    for root, names in (
        (Path('/sys/fs/cgroup'), ('memory.current', 'memory.peak', 'memory.max', 'memory.events')),
        (Path('/sys/fs/cgroup/memory'), ('memory.usage_in_bytes', 'memory.max_usage_in_bytes', 'memory.limit_in_bytes', 'memory.failcnt')),
    ):
        for name in names:
            try:
                value = (root / name).read_text().strip()
                result[name] = ({k: int(v) for k, v in (line.split() for line in value.splitlines())}
                                if name == 'memory.events' else int(value) if value.isdigit() else value)
            except (OSError, ValueError):
                pass
    return result

def diagnostic_event(event: str, **fields) -> None:
    path = os.environ.get(DIAGNOSTICS_ENV, '')
    if not path:
        return
    record = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'monotonic_seconds': time.monotonic(),
        'pid': os.getpid(), 'ppid': os.getppid(),
        'pod': os.environ.get('HOSTNAME', ''),
        'event': event, **memory_snapshot(), **fields,
    }
    # Open/flush each event so a SIGKILL cannot strand Python-buffered records.
    # Locks prevent parent heartbeat and child events interleaving on the PVC.
    with Path(path).open('a', encoding='utf-8') as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        try:
            fp.write(json.dumps(record, allow_nan=False) + '\n')
            fp.flush()
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)

@contextmanager
def diagnostic_stage(stage: str, **fields) -> Iterator[None]:
    start = time.monotonic()
    diagnostic_event('stage_start', stage=stage, **fields)
    try:
        yield
    except BaseException as exc:
        diagnostic_event('stage_failed', stage=stage, elapsed_seconds=time.monotonic() - start,
                         error=repr(exc), traceback=traceback.format_exc(), **fields)
        raise
    else:
        diagnostic_event('stage_complete', stage=stage, elapsed_seconds=time.monotonic() - start, **fields)

@contextmanager
def memory_monitor(interval_seconds: float) -> Iterator[None]:
    stop = threading.Event()
    def monitor() -> None:
        while not stop.wait(interval_seconds):
            diagnostic_event('memory_heartbeat')
    thread = threading.Thread(target=monitor, name='memory-diagnostics', daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()