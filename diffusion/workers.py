"""Safe DataLoader worker counts for diffusion training (fork-after-CUDA guard)."""

from __future__ import annotations

import torch


def num_workers(
    device: torch.device,
    requested: int,
    *,
    allow_fork_after_cuda: bool = False,
) -> int:
    """Return worker count for ``DataLoader``.

    Forking worker processes after the parent has initialized CUDA often segfaults on Linux.
    Default: use ``0`` workers when training on CUDA unless ``allow_fork_after_cuda`` is true.
    """
    rq = int(requested)
    if device.type == "cuda" and rq > 0 and not allow_fork_after_cuda:
        return 0
    return rq
