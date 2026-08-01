from __future__ import annotations
from typing import Any, Dict
import torch
import torch.nn as nn
from nimble.rajagopal_coord_map import RAJAGOPAL_NIMBLE_DOF_NAMES

class ActivationSurrogate(nn.Module):

    def __init__(self, *, input_dim: int=len(RAJAGOPAL_NIMBLE_DOF_NAMES), output_dim: int=80, hidden_dim: int=256, num_layers: int=3, dropout: float=0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        layers: list[nn.Module] = []
        in_d = self.input_dim
        for _ in range(int(num_layers) - 1):
            layers.append(nn.Linear(in_d, int(hidden_dim)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(float(dropout)))
            in_d = int(hidden_dim)
        layers.append(nn.Linear(in_d, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        if q.ndim != 3:
            raise ValueError(f'Expected q [B, T, D], got {tuple(q.shape)}')
        b, t_len, d = q.shape
        x = self.net(q.reshape(b * t_len, d))
        x = x.reshape(b, t_len, self.output_dim)
        return torch.sigmoid(x)

class TransformerActivationSurrogate(nn.Module):

    def __init__(self, *, input_dim: int=len(RAJAGOPAL_NIMBLE_DOF_NAMES), output_dim: int=80, d_model: int=128, num_layers: int=3, num_heads: int=4, dim_feedforward: int=128, dropout: float=0.1, max_seq_len: int=196):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.d_model = int(d_model)
        if self.d_model % int(num_heads) != 0:
            raise ValueError(f'd_model {self.d_model} must be divisible by num_heads {num_heads}')
        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, int(max_seq_len), self.d_model))
        layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=int(num_heads), dim_feedforward=int(dim_feedforward), dropout=float(dropout), batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.fc = nn.Linear(self.d_model, self.output_dim)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        if q.ndim != 3:
            raise ValueError(f'Expected q [B, T, D], got {tuple(q.shape)}')
        t_len = int(q.shape[1])
        if t_len > self.positional_encoding.shape[1]:
            raise ValueError(f'Sequence length {t_len} exceeds max_seq_len {self.positional_encoding.shape[1]}')
        x = self.input_proj(q) + self.positional_encoding[:, :t_len, :]
        x = self.encoder(x)
        return torch.sigmoid(self.fc(x))

def build_activation_surrogate(*, model_type: str='mlp', input_dim: int=len(RAJAGOPAL_NIMBLE_DOF_NAMES), output_dim: int=80, hidden_dim: int=256, num_layers: int=3, dropout: float=0.1, num_heads: int=4, dim_feedforward: int=128, max_seq_len: int=196) -> nn.Module:
    kind = str(model_type).strip().lower()
    if kind == 'mlp':
        return ActivationSurrogate(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    if kind == 'transformer':
        return TransformerActivationSurrogate(input_dim=input_dim, output_dim=output_dim, d_model=int(dim_feedforward), num_layers=num_layers, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, max_seq_len=max_seq_len)
    raise ValueError(f'Unknown surrogate model_type: {model_type!r}')