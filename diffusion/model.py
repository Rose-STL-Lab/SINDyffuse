from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * -emb)
        emb = x.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=1)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        inner = int(dim) * int(mult)
        self.net = nn.Sequential(
            nn.Linear(int(dim), inner),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(inner, int(dim)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MotionDiffusionBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(int(model_dim))
        self.self_attn = nn.MultiheadAttention(int(model_dim), int(num_heads), dropout=float(dropout), batch_first=True)
        self.norm2 = nn.LayerNorm(int(model_dim))
        self.cross_attn = nn.MultiheadAttention(int(model_dim), int(num_heads), dropout=float(dropout), batch_first=True)
        self.norm3 = nn.LayerNorm(int(model_dim))
        self.ff = FeedForward(int(model_dim), mult=4, dropout=float(dropout))

    def forward(self, x: torch.Tensor, text_ctx: torch.Tensor, text_key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + h
        h = self.norm2(x)
        h, _ = self.cross_attn(h, text_ctx, text_ctx, key_padding_mask=text_key_padding_mask, need_weights=False)
        x = x + h
        x = x + self.ff(self.norm3(x))
        return x


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 263,
        model_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        clip_dim: int = 512,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.max_seq_len = int(max_seq_len)
        self.in_proj = nn.Sequential(nn.Linear(self.input_dim, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, self.max_seq_len, model_dim) * 0.02)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.text_proj = nn.Sequential(nn.Linear(clip_dim, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.blocks = nn.ModuleList([MotionDiffusionBlock(model_dim, num_heads, dropout) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, self.input_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_ctx: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        _, tlen, _ = x_t.shape
        if tlen > self.max_seq_len:
            raise ValueError(f"sequence length {tlen} exceeds max_seq_len={self.max_seq_len}")
        if text_ctx is None:
            raise ValueError("text_ctx must be provided")
        h = self.in_proj(x_t) + self.pos_emb[:, :tlen, :]
        h = h + self.time_mlp(t).unsqueeze(1)
        txt = self.text_proj(text_ctx)
        if force_uncond:
            txt = torch.zeros_like(txt)
            kpm = None
        else:
            kpm = (~text_mask.bool()) if text_mask is not None else None
        for blk in self.blocks:
            h = blk(h, txt, kpm)
        return self.out_proj(self.out_norm(h))


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = int(timesteps) + 1
    x = torch.linspace(0, int(timesteps), steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / int(timesteps)) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-8, 0.999).float()


class GaussianDiffusionSchedule:
    def __init__(self, timesteps: int = 1000):
        betas = cosine_beta_schedule(int(timesteps))
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
        self.timesteps = int(timesteps)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def to(self, device: torch.device) -> "GaussianDiffusionSchedule":
        for name in [
            "betas",
            "alphas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
        ]:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def extract(self, arr: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = arr.gather(-1, t).float()
        return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))

