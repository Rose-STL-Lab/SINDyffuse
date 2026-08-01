from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

class _PreNormResidualBlock(nn.Module):

    def __init__(self, dim: int, dropout: float, ff_mult: int=4):
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))
        hidden = int(dim) * int(ff_mult)
        self.ff = nn.Sequential(nn.Linear(int(dim), hidden), nn.GELU(), nn.Dropout(float(dropout)), nn.Linear(hidden, int(dim)), nn.Dropout(float(dropout)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))

class TextToXi(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, theta_dim: int, target_dim: int, num_experts: int=1, max_seq_len: int=256, num_layers: int=3, dropout: float=0.1, ff_mult: int=4):
        super().__init__()
        self.theta_dim = int(theta_dim)
        self.target_dim = int(target_dim)
        self.num_experts = int(max(1, num_experts))
        self.max_seq_len = int(max(1, max_seq_len))
        self.num_layers = int(max(1, num_layers))
        self.dropout_p = float(dropout)
        self.ff_mult = int(ff_mult)
        self.text_encoder = nn.Sequential(nn.Linear(int(in_dim), int(hidden_dim)), nn.LayerNorm(int(hidden_dim)), nn.GELU(), nn.Dropout(self.dropout_p))
        self.frame_pos = nn.Parameter(torch.zeros(self.max_seq_len, int(hidden_dim)))
        nn.init.normal_(self.frame_pos, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList([_PreNormResidualBlock(int(hidden_dim), self.dropout_p, self.ff_mult) for _ in range(self.num_layers)])
        self.out_norm = nn.LayerNorm(int(hidden_dim))
        out_features = self.theta_dim * self.target_dim * self.num_experts
        self.xi_head = nn.Linear(int(hidden_dim), out_features)
        if self.num_experts > 1:
            self.gate_encoder = nn.Sequential(nn.Linear(int(in_dim), int(hidden_dim)), nn.LayerNorm(int(hidden_dim)), nn.GELU(), nn.Dropout(self.dropout_p))
            self.gate_pos = nn.Parameter(torch.zeros(self.max_seq_len, self.num_experts))
            self.gate_head = nn.Linear(int(hidden_dim), self.num_experts)
            nn.init.normal_(self.gate_pos, mean=0.0, std=0.02)

    def forward(self, emb: torch.Tensor, seq_len: Optional[int]=None):
        if seq_len is None:
            raise ValueError('seq_len must be provided for per-frame Xi')
        length = int(seq_len)
        if length > self.max_seq_len:
            raise ValueError(f'seq_len={length} exceeds max_seq_len={self.max_seq_len}')
        text_h = self.text_encoder(emb)
        h = text_h.unsqueeze(1).expand(-1, length, -1) + self.frame_pos[:length, :].unsqueeze(0)
        for block in self.blocks:
            h = block(h)
        h = self.out_norm(h)
        x = self.xi_head(h)
        x = x.view(emb.shape[0], length, self.num_experts, self.theta_dim, self.target_dim)
        if self.num_experts == 1:
            return x[:, :, 0]
        gate_h = self.gate_encoder(emb).unsqueeze(1).expand(-1, length, -1) + self.gate_pos[:length, :].unsqueeze(0)
        gate_logits = self.gate_head(gate_h)
        gates = F.softmax(gate_logits, dim=-1)
        return (x, gates)

def load_text_to_xi_from_checkpoint(ckpt: Dict[str, object]) -> TextToXi:
    model = TextToXi(in_dim=int(ckpt['in_dim']), hidden_dim=int(ckpt['hidden_dim']), theta_dim=int(ckpt['theta_dim']), target_dim=int(ckpt['target_dim']), num_experts=int(ckpt.get('num_experts', 1)), max_seq_len=int(ckpt.get('max_seq_len', 256)), num_layers=int(ckpt['num_layers']), dropout=float(ckpt['dropout']), ff_mult=int(ckpt.get('ff_mult', 4)))
    state = ckpt['model_state']
    if not isinstance(state, dict):
        raise TypeError('checkpoint model_state must be a state dict')
    model.load_state_dict(state)
    return model

def predict_from_xi(theta: torch.Tensor, xi_out: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    if isinstance(xi_out, tuple):
        xi, gates = xi_out
        if xi.dim() == 4:
            pred_k = torch.einsum('blf,bkfz->blkz', theta, xi)
        elif xi.dim() == 5:
            pred_k = torch.einsum('blf,blkfz->blkz', theta, xi)
        else:
            raise ValueError(f'Unsupported xi rank: {xi.dim()}')
        return torch.sum(pred_k * gates.unsqueeze(-1), dim=2)
    if xi_out.dim() == 4:
        return torch.einsum('blf,blfz->blz', theta, xi_out)
    return torch.einsum('blf,bfz->blz', theta, xi_out)

def xi_temporal_smoothness(xi_out: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    xi_tensor = xi_out[0] if isinstance(xi_out, tuple) else xi_out
    if xi_tensor.dim() not in (4, 5) or xi_tensor.shape[1] < 3:
        return xi_tensor.new_tensor(0.0)
    d2 = xi_tensor[:, 2:] - 2.0 * xi_tensor[:, 1:-1] + xi_tensor[:, :-2]
    return torch.mean(d2 ** 2)