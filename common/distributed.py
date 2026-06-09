"""PyTorch distributed (DDP) helpers for SINDyffuse training."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

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

    use_backend = str((distributed_cfg or {}).get("backend", backend)).strip().lower()
    if use_backend == "nccl" and not torch.cuda.is_available():
        use_backend = "gloo"

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=use_backend, rank=rank, world_size=world_size)

    _DIST_INITIALIZED = True
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
