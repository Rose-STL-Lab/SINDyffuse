"""Text-conditioned sparse coefficients Xi for SINDy guidance."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextToXi(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        theta_dim: int,
        target_dim: int,
        num_experts: int = 1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.theta_dim = int(theta_dim)
        self.target_dim = int(target_dim)
        self.num_experts = int(max(1, num_experts))
        self.max_seq_len = int(max(1, max_seq_len))
        self.frame_proj = nn.Linear(int(in_dim), int(hidden_dim))
        self.frame_pos = nn.Parameter(torch.zeros(self.max_seq_len, int(hidden_dim)))
        self.frame_head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(theta_dim) * int(target_dim) * self.num_experts),
        )
        nn.init.normal_(self.frame_pos, mean=0.0, std=0.02)
        if self.num_experts > 1:
            self.gate_text = nn.Sequential(
                nn.Linear(int(in_dim), int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), self.num_experts),
            )
            self.gate_pos = nn.Parameter(torch.zeros(self.max_seq_len, self.num_experts))
            nn.init.normal_(self.gate_pos, mean=0.0, std=0.02)

    def forward(self, emb: torch.Tensor, seq_len: Optional[int] = None):
        if seq_len is None:
            raise ValueError("seq_len must be provided for per-frame Xi")
        length = int(seq_len)
        if length > self.max_seq_len:
            raise ValueError(f"seq_len={length} exceeds max_seq_len={self.max_seq_len}")
        h = self.frame_proj(emb).unsqueeze(1) + self.frame_pos[:length, :].unsqueeze(0)
        h = F.gelu(h)
        x = self.frame_head(h)
        x = x.view(emb.shape[0], length, self.num_experts, self.theta_dim, self.target_dim)
        if self.num_experts == 1:
            return x[:, :, 0]
        gate_logits = self.gate_text(emb).unsqueeze(1) + self.gate_pos[:length, :].unsqueeze(0)
        gates = F.softmax(gate_logits, dim=-1)
        return x, gates


def predict_from_xi(theta: torch.Tensor, xi_out: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    if isinstance(xi_out, tuple):
        xi, gates = xi_out
        if xi.dim() == 4:
            pred_k = torch.einsum("blf,bkfz->blkz", theta, xi)
        elif xi.dim() == 5:
            pred_k = torch.einsum("blf,blkfz->blkz", theta, xi)
        else:
            raise ValueError(f"Unsupported xi rank: {xi.dim()}")
        return torch.sum(pred_k * gates.unsqueeze(-1), dim=2)
    if xi_out.dim() == 4:
        return torch.einsum("blf,blfz->blz", theta, xi_out)
    return torch.einsum("blf,bfz->blz", theta, xi_out)


def xi_temporal_smoothness(xi_out: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    xi_tensor = xi_out[0] if isinstance(xi_out, tuple) else xi_out
    if xi_tensor.dim() not in (4, 5) or xi_tensor.shape[1] < 3:
        return xi_tensor.new_tensor(0.0)
    d2 = xi_tensor[:, 2:] - 2.0 * xi_tensor[:, 1:-1] + xi_tensor[:, :-2]
    return torch.mean(d2**2)
