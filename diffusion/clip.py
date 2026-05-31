"""CLIP text token embeddings for diffusion conditioning."""

from __future__ import annotations

from typing import List, Tuple

import torch

import clip  # type: ignore


@torch.no_grad()
def clip_encode(
    clip_model: torch.nn.Module,
    captions: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    toks = clip.tokenize(captions, truncate=True).to(device)
    x = clip_model.token_embedding(toks).type(clip_model.dtype)
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)
    x = clip_model.ln_final(x).float()
    return x, toks != 0
