"""Unit tests for evaluation metrics and protocol."""

from __future__ import annotations

import unittest

import numpy as np

from eval.metrics import (
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_matching_score,
    calculate_top_k,
    euclidean_distance_matrix,
)


class MetricsTests(unittest.TestCase):
    def test_euclidean_distance_matrix(self):
        a = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        dist = euclidean_distance_matrix(a, b)
        self.assertAlmostEqual(dist[0, 0], 0.0, places=5)
        self.assertAlmostEqual(dist[1, 1], np.sqrt(2.0), places=5)

    def test_top_k_identity(self):
        mat = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]])
        top1 = calculate_top_k(mat, top_k=1)
        self.assertTrue(np.all(top1[:, 0]))

    def test_matching_score_zero_for_identical(self):
        emb = np.random.randn(8, 16).astype(np.float32)
        score = calculate_matching_score(emb, emb)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_fid_zero_for_same_distribution(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(64, 32)).astype(np.float32)
        mu1, cov1 = calculate_activation_statistics(x)
        mu2, cov2 = calculate_activation_statistics(x)
        fid = calculate_frechet_distance(mu1, cov1, mu2, cov2)
        self.assertAlmostEqual(fid, 0.0, places=4)

    def test_diversity_positive(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(64, 32)).astype(np.float32)
        div = calculate_diversity(x, diversity_times=32)
        self.assertGreater(div, 0.0)


class EvaluatorSmokeTests(unittest.TestCase):
    def test_evaluator_checkpoint_exists(self):
        from eval.config import default_evaluator_root

        root = default_evaluator_root()
        ckpt = root / "t2m" / "text_mot_match" / "model" / "finest.tar"
        if not ckpt.is_file():
            self.skipTest(f"Evaluator checkpoint not found at {ckpt}")
        self.assertTrue(ckpt.is_file())

    def test_word_vectorizer_loads(self):
        from eval.config import default_evaluator_root
        from eval.word_vectorizer import WordVectorizer

        root = default_evaluator_root()
        glove = root / "glove"
        if not (glove / "our_vab_data.npy").is_file():
            self.skipTest("GloVe vectors not found")
        vec = WordVectorizer(glove)
        emb, pos = vec["walk/VERB"]
        self.assertEqual(emb.shape[0], 300)
        self.assertEqual(pos.shape[0], 15)


if __name__ == "__main__":
    unittest.main()
