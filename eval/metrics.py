"""HumanML3D evaluation metrics (FID, R-Precision, MM-Dist, Diversity).

Adapted from the T2M / StableMoFusion evaluation protocol:
https://github.com/EricGuo5513/text-to-motion
"""

from __future__ import annotations

import numpy as np
from scipy import linalg


def euclidean_distance_matrix(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)
    d3 = np.sum(np.square(matrix2), axis=1)
    return np.sqrt(np.maximum(d1 + d2 + d3, 0.0))


def calculate_top_k(mat: np.ndarray, top_k: int) -> np.ndarray:
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = mat == gt_mat
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = correct_vec | bool_mat[:, i]
        top_k_list.append(correct_vec[:, None])
    return np.concatenate(top_k_list, axis=1)


def calculate_R_precision(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    top_k: int,
    *,
    sum_all: bool = False,
) -> np.ndarray:
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    return top_k_mat


def calculate_matching_score(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    *,
    sum_all: bool = False,
) -> float | np.ndarray:
    assert embedding1.shape == embedding2.shape
    dist = linalg.norm(embedding1 - embedding2, axis=1)
    if sum_all:
        return dist.sum(axis=0)
    return float(dist.mean())


def calculate_activation_statistics(activations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation: np.ndarray, diversity_times: int) -> float:
    assert activation.ndim == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return float(dist.mean())


def calculate_multimodality(activation: np.ndarray, multimodality_times: int) -> float:
    assert activation.ndim == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]
    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return float(dist.mean())


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
