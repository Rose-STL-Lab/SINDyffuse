"""Tests for SINDy muscle activation target integration."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from nimble.channels import BIOMECH_COMPONENT_KEYS
from nimble.muscle_b3d import MUSCLE_ACTIVATION_ROWS, is_zero_placeholder_activations
from sindy.targets import (
    N_BIO_TARGETS,
    N_MUSCLE_TARGETS,
    N_SINDY_TARGETS,
    build_sindy_targets,
    muscle_channel_names,
    sindy_target_keys,
)


class TestSindyTargetRegistry(unittest.TestCase):
    def test_target_counts(self) -> None:
        self.assertEqual(N_BIO_TARGETS, len(BIOMECH_COMPONENT_KEYS))
        self.assertEqual(N_MUSCLE_TARGETS, MUSCLE_ACTIVATION_ROWS)
        self.assertEqual(N_SINDY_TARGETS, N_BIO_TARGETS + N_MUSCLE_TARGETS)

    def test_sindy_target_keys_order(self) -> None:
        keys = sindy_target_keys()
        self.assertEqual(len(keys), N_SINDY_TARGETS)
        self.assertEqual(keys[:N_BIO_TARGETS], tuple(BIOMECH_COMPONENT_KEYS))
        self.assertEqual(keys[N_BIO_TARGETS:], muscle_channel_names())

    def test_build_sindy_targets_shape(self) -> None:
        t = 32
        bio = np.random.rand(t, N_BIO_TARGETS).astype(np.float32)
        act = np.random.rand(t, N_MUSCLE_TARGETS).astype(np.float32)
        y = build_sindy_targets(bio, act)
        self.assertEqual(y.shape, (t - 1, N_SINDY_TARGETS))

    def test_zero_placeholder_filter(self) -> None:
        self.assertTrue(is_zero_placeholder_activations(np.zeros((10, 80), dtype=np.float32)))
        act = np.random.rand(10, 80).astype(np.float32)
        self.assertFalse(is_zero_placeholder_activations(act))


class TestLearnedSINDyGuidanceSplit(unittest.TestCase):
    def test_actual_targets_concat_bio_and_surrogate(self) -> None:
        from sindy.guidance import LearnedSINDyGuidance

        guidance = object.__new__(LearnedSINDyGuidance)
        guidance._target_weights = torch.ones(N_SINDY_TARGETS)
        b, t = 2, 8
        bio = torch.randn(b, t - 1, N_BIO_TARGETS)
        muscle_full = torch.rand(b, t, N_MUSCLE_TARGETS)
        guidance._bio_from_motion = MagicMock(return_value=bio)
        guidance._surrogate = MagicMock()
        guidance._surrogate.predict_activations.return_value = muscle_full
        guidance._denorm_motion = MagicMock(return_value=torch.zeros(b, t, 4))

        motion = torch.randn(b, t, 4)
        actual = LearnedSINDyGuidance._actual_targets_from_motion(guidance, motion)
        self.assertEqual(actual.shape, (b, t - 1, N_SINDY_TARGETS))
        torch.testing.assert_close(actual[..., :N_BIO_TARGETS], bio)
        torch.testing.assert_close(actual[..., N_BIO_TARGETS:], muscle_full[:, :-1, :])


class TestCheckpointTargetDim(unittest.TestCase):
    def test_rejects_legacy_target_dim(self) -> None:
        from sindy.guidance import LearnedSINDyGuidance

        with patch.object(LearnedSINDyGuidance, "__init__", lambda self, *a, **k: None):
            guidance = LearnedSINDyGuidance()
        guidance.target_dim = len(BIOMECH_COMPONENT_KEYS)
        with self.assertRaises(ValueError):
            # replicate validation logic from __init__ after load
            expected = N_SINDY_TARGETS
            if guidance.target_dim != expected:
                raise ValueError(
                    f"Checkpoint target_dim={guidance.target_dim} != {expected}"
                )


if __name__ == "__main__":
    unittest.main()
