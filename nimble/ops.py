from __future__ import annotations
import torch

def finite_diff(x: torch.Tensor, dt: float) -> torch.Tensor:
    if x.shape[0] < 2:
        return torch.zeros_like(x)
    y = torch.zeros_like(x)
    y[0] = (x[1] - x[0]) / float(dt)
    y[-1] = (x[-1] - x[-2]) / float(dt)
    if x.shape[0] > 2:
        y[1:-1] = (x[2:] - x[:-2]) / (2.0 * float(dt))
    return y

def safe_norm(x: torch.Tensor, dim: int=1) -> torch.Tensor:
    return torch.linalg.norm(x, dim=dim, keepdim=True)

def sanitize_joint_positions_torch(joints: torch.Tensor, *, clamp_m: float=50.0) -> torch.Tensor:
    j = torch.nan_to_num(joints, nan=0.0, posinf=float(clamp_m), neginf=-float(clamp_m))
    return torch.clamp(j, min=-float(clamp_m), max=float(clamp_m))