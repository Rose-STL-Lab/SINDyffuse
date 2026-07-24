"""PyTorch distributed (DDP) helpers for SINDyffuse training."""

from __future__ import annotations

import glob
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from common.runtime import resolve_torch_device, set_seed

_DIST_INITIALIZED = False


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    return int(raw)


def torchrun_launched() -> bool:
    return _env_int("WORLD_SIZE", 1) > 1


def under_torchrun() -> bool:
    """True when this process was started by ``torchrun`` / ``torch.distributed.run``."""
    if _env_int("WORLD_SIZE", 1) > 1:
        return True
    return os.environ.get("LOCAL_RANK", "").strip() != ""


def cuda_usable(device_index: int = 0) -> bool:
    """True when CUDA kernels can run on ``device_index`` (not just ``is_available()``)."""
    if not torch.cuda.is_available():
        return False
    if device_index < 0 or device_index >= torch.cuda.device_count():
        return False
    try:
        torch.cuda.set_device(device_index)
        x = torch.randn(2, 2, device=f"cuda:{device_index}")
        _ = float(x.sum())
        return True
    except RuntimeError:
        return False


def usable_cuda_device_count() -> int:
    """Count CUDA devices that can actually run kernels."""
    if not torch.cuda.is_available():
        return 0
    return sum(1 for i in range(torch.cuda.device_count()) if cuda_usable(i))


def _requested_nproc_per_node(*, cap: Optional[int] = None) -> int:
    raw = os.environ.get("NPROC_PER_NODE", "").strip()
    if raw:
        n = max(1, int(raw))
        if cap is not None:
            n = min(n, int(cap))
        return n

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        devs = [d.strip() for d in visible.split(",") if d.strip()]
        if devs:
            n = len(devs)
            if cap is not None:
                n = min(n, int(cap))
            return max(1, n)

    if torch.cuda.is_available():
        n = int(torch.cuda.device_count())
        if cap is not None:
            n = min(n, int(cap))
        return max(1, n)

    return 1


def resolve_nproc_per_node(*, cap: Optional[int] = None) -> int:
    """Return GPU/process count for multi-GPU training (always >= 1).

    Resolution order:
    1. ``NPROC_PER_NODE`` env (matches ``nvidia.com/gpu`` in K8s when set)
    2. ``CUDA_VISIBLE_DEVICES`` entry count
    3. ``torch.cuda.device_count()`` when CUDA is available
    4. ``1`` (CPU / no GPU)

    The result is capped by the number of CUDA devices that can run kernels.
    """
    requested = _requested_nproc_per_node(cap=cap)
    usable = usable_cuda_device_count()
    if usable <= 0:
        return requested

    if requested > usable:
        print(
            f"[distributed] Requested nproc_per_node={requested} but only {usable} "
            f"usable CUDA device(s) are visible; using {usable}.",
            flush=True,
        )
        return usable
    return requested


def _cuda_env_summary(*, local_rank: Optional[int] = None) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    nproc = os.environ.get("NPROC_PER_NODE", "unset")
    dev_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    usable = usable_cuda_device_count()
    nodes = sorted(glob.glob("/dev/nvidia[0-9]*"))
    parts = [
        f"NPROC_PER_NODE={nproc}",
        f"CUDA_VISIBLE_DEVICES={visible}",
        f"torch.cuda.device_count={dev_count}",
        f"usable_cuda_devices={usable}",
        f"nvidia_device_nodes={nodes or 'none'}",
    ]
    if local_rank is not None:
        parts.append(f"local_rank={local_rank}")
    return " ".join(parts)


def _raise_cuda_unusable(*, local_rank: int, world_size: int) -> None:
    raise RuntimeError(
        "CUDA is visible but kernels fail on this process "
        f"(rank/local_rank={local_rank}, world_size={world_size}). "
        f"{_cuda_env_summary(local_rank=local_rank)}. "
        "Common causes in Kubernetes: the pod was scheduled without enough GPUs, "
        "nvidia.com/gpu limits do not match NPROC_PER_NODE, or the NVIDIA device "
        "plugin did not mount /dev/nvidia* inside the container. "
        "Recreate the job on a GPU node, align nvidia.com/gpu with NPROC_PER_NODE, "
        "or set NPROC_PER_NODE=1 for single-GPU training."
    )


def _torchrun_executable() -> List[str]:
    """Return argv prefix to invoke ``torchrun`` (or ``python -m torch.distributed.run``)."""
    path = shutil.which("torchrun")
    if path:
        return [path]
    return [sys.executable, "-m", "torch.distributed.run"]


def maybe_relaunch_with_torchrun(*, module: Optional[str] = None) -> None:
    """Re-exec the current training entry point under ``torchrun`` when needed.

    No-op when already launched by torchrun, when ``NPROC_PER_NODE=1``, or when only
    one process should run. Call near the top of training ``main()`` after parsing
    CLI args (and optionally checking ``parse_distributed_enabled``).
    """
    if under_torchrun():
        return
    if os.environ.get("SINDYFFUSE_NO_TORCHRUN", "").strip().lower() in {"1", "true", "yes"}:
        return

    nproc = resolve_nproc_per_node()
    if nproc <= 1:
        return

    usable = usable_cuda_device_count()
    if torch.cuda.is_available() and usable <= 0:
        _raise_cuda_unusable(local_rank=0, world_size=nproc)

    prefix = _torchrun_executable()
    cmd = [*prefix, "--standalone", f"--nproc_per_node={nproc}"]
    if module:
        cmd.extend(["-m", module, *sys.argv[1:]])
    else:
        cmd.extend([sys.argv[0], *sys.argv[1:]])

    print(
        f"[distributed] Relaunching under torchrun (nproc_per_node={nproc}): "
        f"{' '.join(cmd)}",
        flush=True,
    )
    os.execv(cmd[0], cmd)


def parse_distributed_enabled(cfg: Optional[Dict[str, Any]]) -> bool:
    """Resolve ``distributed.enabled``: auto | true | false."""
    if not cfg:
        return torchrun_launched()
    raw = str(cfg.get("enabled", "auto")).strip().lower()
    if raw in {"true", "1", "yes", "on"}:
        return True
    if raw in {"false", "0", "no", "off"}:
        return False
    return torchrun_launched()


def distributed_explicitly_disabled(cfg: Optional[Dict[str, Any]]) -> bool:
    if not cfg:
        return False
    raw = str(cfg.get("enabled", "auto")).strip().lower()
    return raw in {"false", "0", "no", "off"}


def should_auto_relaunch_torchrun(cfg: Optional[Dict[str, Any]] = None) -> bool:
    """True when training should re-exec under torchrun before DDP init."""
    if under_torchrun():
        return False
    if distributed_explicitly_disabled(cfg):
        return False
    if os.environ.get("SINDYFFUSE_NO_TORCHRUN", "").strip().lower() in {"1", "true", "yes"}:
        return False
    return resolve_nproc_per_node() > 1


def init_distributed(
    *,
    backend: str = "nccl",
    distributed_cfg: Optional[Dict[str, Any]] = None,
) -> bool:
    """Initialize process group when enabled. Returns whether DDP is active."""
    global _DIST_INITIALIZED
    if not parse_distributed_enabled(distributed_cfg):
        return False
    if _DIST_INITIALIZED:
        return True

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    local_rank = _env_int("LOCAL_RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)

    if world_size <= 1:
        return False

    log_gpu_diagnostics_preinit()

    use_backend = str((distributed_cfg or {}).get("backend", backend)).strip().lower()
    usable = usable_cuda_device_count()
    if use_backend == "nccl":
        if local_rank >= usable:
            _raise_cuda_unusable(local_rank=local_rank, world_size=world_size)
        if not cuda_usable(local_rank):
            if torch.cuda.is_available():
                _raise_cuda_unusable(local_rank=local_rank, world_size=world_size)
            use_backend = "gloo"
        elif world_size > usable:
            _raise_cuda_unusable(local_rank=local_rank, world_size=world_size)

    if use_backend == "nccl":
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=use_backend, rank=rank, world_size=world_size)

    _DIST_INITIALIZED = True
    log_gpu_diagnostics()
    return True


def is_distributed() -> bool:
    return _DIST_INITIALIZED and dist.is_initialized() and get_world_size() > 1


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return _env_int("RANK", 0)


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size())
    return _env_int("WORLD_SIZE", 1)


def get_local_rank() -> int:
    return _env_int("LOCAL_RANK", 0)


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def cleanup_distributed() -> None:
    global _DIST_INITIALIZED
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    _DIST_INITIALIZED = False


def resolve_train_device(device_name: str = "auto") -> torch.device:
    if is_distributed() and torch.cuda.is_available():
        return torch.device(f"cuda:{get_local_rank()}")
    return resolve_torch_device(device_name)


def seed_all(seed: int) -> None:
    set_seed(int(seed) + get_rank())


def wrap_ddp(
    model: torch.nn.Module,
    *,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    if not is_distributed():
        return model
    device = resolve_train_device("auto")
    model = model.to(device)
    return DDP(
        model,
        device_ids=[get_local_rank()] if device.type == "cuda" else None,
        output_device=get_local_rank() if device.type == "cuda" else None,
        find_unused_parameters=bool(find_unused_parameters),
    )


def unwrap_module(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def model_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return unwrap_module(model).state_dict()


def shard_indices(indices: Union[torch.Tensor, Any], rank: Optional[int] = None) -> Any:
    """Shard a 1-D index array across ranks (strided split)."""
    r = get_rank() if rank is None else int(rank)
    ws = get_world_size()
    return indices[r::ws]


def setup_spawn_if_distributed() -> None:
    if _env_int("WORLD_SIZE", 1) <= 1:
        return
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass


def log_main(msg: str) -> None:
    if is_main_process():
        from common.run_logging import get_run_logger

        get_run_logger().verbose(msg)


def log_gpu_diagnostics_preinit() -> None:
    """Log GPU env before DDP init (rank 0 only; uses print so it always appears)."""
    if get_local_rank() != 0:
        return
    print(f"[distributed/gpu] preinit {_cuda_env_summary()}", flush=True)


def log_gpu_diagnostics() -> None:
    """Log visible GPUs and distributed env (every rank; cheap sanity check)."""
    if not is_main_process():
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    nproc = os.environ.get("NPROC_PER_NODE", "")
    world_env = os.environ.get("WORLD_SIZE", "")
    cuda_ok = torch.cuda.is_available()
    dev_count = int(torch.cuda.device_count()) if cuda_ok else 0
    usable = usable_cuda_device_count()
    log_main(
        "[distributed/gpu] "
        f"NPROC_PER_NODE={nproc or 'unset'} "
        f"CUDA_VISIBLE_DEVICES={visible or 'unset'} "
        f"WORLD_SIZE_env={world_env or 'unset'} "
        f"rank={get_rank()} local_rank={get_local_rank()} "
        f"world_size={get_world_size()} distributed={is_distributed()} "
        f"torch.cuda.is_available={cuda_ok} torch.cuda.device_count={dev_count} "
        f"usable_cuda_devices={usable}"
    )
