"""CPU detection and BLAS/OpenMP thread configuration for heavy numerical work."""

from __future__ import annotations

import os
from pathlib import Path


def _cgroup_v2_cpu_count() -> int | None:
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if not cpu_max.is_file():
        return None
    try:
        line = cpu_max.read_text(encoding="utf-8").strip().split()
        if len(line) < 2:
            return None
        quota, period = line[0], line[1]
        if quota == "max":
            return None
        q, p = int(quota), int(period)
        if q > 0 and p > 0:
            return max(1, q // p)
    except (OSError, ValueError):
        return None
    return None


def _cgroup_v1_cpu_count() -> int | None:
    root = Path("/sys/fs/cgroup/cpu,cpuacct")
    if not root.is_dir():
        root = Path("/sys/fs/cgroup")
    quota_path = root / "cpu.cfs_quota_us"
    period_path = root / "cpu.cfs_period_us"
    if not quota_path.is_file() or not period_path.is_file():
        return None
    try:
        quota = int(quota_path.read_text(encoding="utf-8").strip())
        period = int(period_path.read_text(encoding="utf-8").strip())
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        return None
    return None


def detect_usable_cpus() -> int:
    """Usable CPUs for this process (cgroup quota, affinity mask, or host count)."""
    env = os.environ.get("MOCO_NUM_THREADS", "").strip()
    if env.isdigit():
        return max(1, int(env))

    try:
        affinity = len(os.sched_getaffinity(0))
        if affinity > 0:
            return affinity
    except (AttributeError, NotImplementedError, OSError):
        pass

    for detector in (_cgroup_v2_cpu_count, _cgroup_v1_cpu_count):
        n = detector()
        if n is not None and n > 0:
            return n

    return max(1, int(os.cpu_count() or 1))


def configure_compute_threads(num_threads: int) -> int:
    """Set OpenMP/BLAS thread env vars; returns the count applied."""
    n = max(1, int(num_threads))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(n)
    return n


def resolve_k8s_shard(
    *,
    num_shards: int | None = None,
    shard_index: int | None = None,
) -> tuple[int, int]:
    """Return ``(shard_index, num_shards)`` for distributed preprocess.

    Defaults to ``(0, 1)`` (no sharding). When ``num_shards > 1``, ``shard_index``
    is taken from ``shard_index`` or ``JOB_COMPLETION_INDEX``.
    """
    env_shards = os.environ.get("PREPROCESS_NUM_SHARDS", "").strip()
    n = int(num_shards) if num_shards is not None else (int(env_shards) if env_shards.isdigit() else 1)
    if n <= 1:
        return 0, 1

    if shard_index is not None:
        i = int(shard_index)
    else:
        idx_env = os.environ.get("JOB_COMPLETION_INDEX", "").strip()
        if not idx_env.isdigit():
            raise ValueError(
                "num_shards > 1 requires --shard_index or JOB_COMPLETION_INDEX (Indexed Job)"
            )
        i = int(idx_env)
    if i < 0 or i >= n:
        raise ValueError(f"shard_index must be in [0, {n}), got {i}")
    return i, n


def resolve_preprocess_parallelism(
    num_workers: int,
    *,
    activation_method: str = "moco_track",
    skip_muscle_activation: bool = False,
    moco_parallel_motions: int = 1,
    num_shards: int = 1,
) -> tuple[int, int]:
    """Return ``(motion_process_workers, opensim_thread_count)``.

    ``none``: parallel IK-only workers (no OpenSim).

    ``static_optimization``: one motion at a time, 1 OpenSim thread (memory-heavy).

    ``moco_track``: concurrent Moco solves; threads per solve from ``num_workers``.

    When ``num_shards > 1`` (K8s distributed), in-pod parallelism is 1 for all methods;
    cluster-level parallelism comes from the shard count.
    """
    if int(num_shards) > 1:
        return 1, 1

    auto = detect_usable_cpus()
    method = "none" if skip_muscle_activation else str(activation_method)
    if method == "none":
        pool = max(1, int(num_workers)) if int(num_workers) > 0 else auto
        return pool, 1
    if method == "static_optimization":
        return 1, 1
    parallel = max(1, int(moco_parallel_motions))
    total = max(1, int(num_workers)) if int(num_workers) > 0 else auto
    threads = max(1, total // parallel)
    return parallel, threads
