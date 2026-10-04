from __future__ import annotations
import torch

def activation_surrogate_loss(pred, target, *, lambda_temporal: float=0.1, valid_mask=None):
    mask = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device) if valid_mask is None else valid_mask.bool()
    if not mask.any():
        raise ValueError('Batch has no valid activation frames')
    loss = torch.abs(pred[mask] - target[mask]).mean()
    if pred.shape[1] >= 2 and lambda_temporal > 0:
        adjacent = mask[:, 1:] & mask[:, :-1]
        if adjacent.any():
            dp = pred[:, 1:][adjacent] - pred[:, :-1][adjacent]
            dt = target[:, 1:][adjacent] - target[:, :-1][adjacent]
            loss = loss + float(lambda_temporal) * torch.abs(dp - dt).mean()
    return loss