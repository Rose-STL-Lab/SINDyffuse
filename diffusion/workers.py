from __future__ import annotations
import torch

def num_workers(device: torch.device, requested: int, *, allow_fork_after_cuda: bool=False) -> int:
    rq = int(requested)
    if device.type == 'cuda' and rq > 0 and (not allow_fork_after_cuda):
        return 0
    return rq