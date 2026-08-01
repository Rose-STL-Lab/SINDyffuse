from __future__ import annotations
from typing import Dict, Sequence, Tuple
import numpy as np

def _pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    ab = a @ b.T
    d2 = np.clip(aa + bb - 2.0 * ab, 0.0, None)
    return np.sqrt(d2)

def r_precision(motion_emb: np.ndarray, text_emb: np.ndarray, *, matched_text_idx: Sequence[int] | None=None, pool_size: int=32) -> Dict[str, float]:
    motion_emb = np.asarray(motion_emb, dtype=np.float64)
    text_emb = np.asarray(text_emb, dtype=np.float64)
    n = int(motion_emb.shape[0])
    if matched_text_idx is None:
        matched_text_idx = list(range(n))
    if len(matched_text_idx) != n:
        raise ValueError('matched_text_idx length must match motion count')
    hits = {1: 0, 2: 0, 3: 0}
    for i in range(n):
        pool_n = min(int(pool_size), int(text_emb.shape[0]))
        idx = np.arange(text_emb.shape[0])
        if pool_n < text_emb.shape[0]:
            rng = np.random.default_rng(i)
            idx = rng.choice(text_emb.shape[0], size=pool_n, replace=False)
        if int(matched_text_idx[i]) not in idx:
            idx = np.concatenate([idx, [int(matched_text_idx[i])]])
        pool = text_emb[idx]
        dist = _pairwise_dist(motion_emb[i:i + 1], pool).reshape(-1)
        order = np.argsort(dist)
        ranks = idx[order]
        target = int(matched_text_idx[i])
        pos = int(np.where(ranks == target)[0][0])
        for k in (1, 2, 3):
            if pos < k:
                hits[k] += 1
    return {f'top{k}': hits[k] / max(n, 1) for k in (1, 2, 3)}

def fid(real_emb: np.ndarray, gen_emb: np.ndarray) -> float:
    real_emb = np.asarray(real_emb, dtype=np.float64)
    gen_emb = np.asarray(gen_emb, dtype=np.float64)
    mu_r = real_emb.mean(axis=0)
    mu_g = gen_emb.mean(axis=0)
    sig_r = np.cov(real_emb, rowvar=False)
    sig_g = np.cov(gen_emb, rowvar=False)
    if sig_r.ndim == 0:
        sig_r = np.array([[float(sig_r)]], dtype=np.float64)
    if sig_g.ndim == 0:
        sig_g = np.array([[float(sig_g)]], dtype=np.float64)
    diff = mu_r - mu_g
    covmean = _matrix_sqrt(sig_r @ sig_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig_r + sig_g - 2.0 * covmean))

def _matrix_sqrt(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0.0, None)
    return vecs * np.sqrt(vals) @ vecs.T

def mm_dist(motion_emb: np.ndarray, text_emb: np.ndarray) -> float:
    motion_emb = np.asarray(motion_emb, dtype=np.float64)
    text_emb = np.asarray(text_emb, dtype=np.float64)
    n = min(motion_emb.shape[0], text_emb.shape[0])
    if n == 0:
        return float('nan')
    d = _pairwise_dist(motion_emb[:n], text_emb[:n])
    return float(np.mean(np.diag(d)))

def diversity(motion_emb: np.ndarray) -> float:
    motion_emb = np.asarray(motion_emb, dtype=np.float64)
    n = int(motion_emb.shape[0])
    if n < 2:
        return 0.0
    d = _pairwise_dist(motion_emb, motion_emb)
    tri = d[np.triu_indices(n, k=1)]
    return float(np.mean(tri))

def text_alignment_bundle(motion_emb: np.ndarray, text_emb: np.ndarray, reference_motion_emb: np.ndarray, *, matched_text_idx: Sequence[int] | None=None) -> Dict[str, float]:
    rp = r_precision(motion_emb, text_emb, matched_text_idx=matched_text_idx)
    out = {'r_precision_top1': rp['top1'], 'r_precision_top2': rp['top2'], 'r_precision_top3': rp['top3'], 'fid': fid(reference_motion_emb, motion_emb), 'mm_dist': mm_dist(motion_emb, text_emb), 'diversity': diversity(motion_emb)}
    return out