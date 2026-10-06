"""Immutable indexed motion assignments and atomic per-motion outcomes."""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

def identity(payload: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in payload.items() if k != 'task_set_id'},
                                    sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as fp:
            json.dump(payload, fp, indent=2, allow_nan=False)
            fp.write('\n'); fp.flush(); os.fsync(fp.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)

@contextmanager
def task_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)

def prepare_tasks(directory: Path, *, motion_ids: list[str], configuration: dict,
                  artifacts: dict, expected_tasks: int, data_root: Path, ik_rows: dict | None=None) -> dict:
    if not motion_ids or len(set(motion_ids)) != len(motion_ids):
        raise ValueError('Motion tasks must be nonempty and unique')
    if len(motion_ids) != expected_tasks:
        raise ValueError(f'Job expects {expected_tasks} motions, dataset selection contains {len(motion_ids)}')
    payload = {'schema_version': 1, 'motion_ids': motion_ids, 'task_count': len(motion_ids),
               'configuration': configuration, 'artifacts': artifacts, 'data_root': str(data_root.resolve()),
               'ik_rows': ik_rows or {}}
    payload['task_set_id'] = identity(payload)
    directory.mkdir(parents=True, exist_ok=True)
    with task_lock(directory / '.prepare.lock'):
        path = directory / 'tasks.json'
        if path.exists():
            if load_tasks(path) != payload:
                raise ValueError('Task inputs changed; choose a new MOTION_TASK_DIR instead of replacing assignments')
        else:
            atomic_json(path, payload)
    return payload

def load_tasks(path: Path) -> dict:
    payload = json.loads(path.read_text())
    ids = payload['motion_ids']
    if (payload.get('schema_version') != 1 or payload.get('task_set_id') != identity(payload)
            or payload['task_count'] != len(ids) or not ids or len(set(ids)) != len(ids)):
        raise ValueError('Invalid motion task manifest')
    return payload

def task_motion(payload: dict, index: int, expected_tasks: int) -> str:
    if payload['task_count'] != expected_tasks:
        raise ValueError('Indexed Job completions do not match prepared motion task count')
    if not 0 <= index < expected_tasks:
        raise ValueError(f'Invalid motion task index {index}')
    return payload['motion_ids'][index]

def outcome_path(directory: Path, index: int) -> Path:
    return directory / 'outcomes' / f'{index:06d}.json'

def load_outcomes(directory: Path) -> tuple[dict, list[dict]]:
    tasks = load_tasks(directory / 'tasks.json')
    rows = []
    for index, mid in enumerate(tasks['motion_ids']):
        row = json.loads(outcome_path(directory, index).read_text())
        if row.get('id') != mid or row.get('task_index') != index or row.get('task_set_id') != tasks['task_set_id']:
            raise ValueError(f'Outcome identity mismatch for task {index}')
        if row.get('status') not in ('ok', 'skipped', 'moco_failed', 'moco_skipped'):
            raise ValueError(f'Task {index} has unresolved infrastructure/data error; retry it before normalization')
        rows.append(row)
    return tasks, rows