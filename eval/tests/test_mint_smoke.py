"""Smoke tests for MinT / OpenSim integration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class MintSmokeTests(unittest.TestCase):
    def test_biomech_keys_count(self):
        from common.biomech import BIOMECH_COMPONENT_KEYS

        self.assertEqual(len(BIOMECH_COMPONENT_KEYS), 22)

    def test_cache_schema_roundtrip(self):
        import numpy as np

        from osim.cache_schema import write_motion_cache, read_motion_cache, KEY_Q, KEY_MUSCLE

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "000001.npz"
            q = np.random.randn(20, 33).astype(np.float32)
            muscle = np.random.rand(20, 402).astype(np.float32)
            write_motion_cache(
                path,
                q=q,
                muscle_activations=muscle,
                guidance_features=np.zeros((20, 22), dtype=np.float32),
                has_mint_labels=True,
            )
            data = read_motion_cache(path)
            self.assertEqual(data[KEY_Q].shape, q.shape)
            self.assertEqual(data[KEY_MUSCLE].shape, muscle.shape)

    def test_retarget_import(self):
        from osim.retarget import retarget_hml_joints_to_q

        self.assertTrue(callable(retarget_hml_joints_to_q))

    def test_marker_schema(self):
        from osim.ssm67_markers import MARKER_NAMES, ensure_ssm67_marker_xml
        from osim.trc_io import write_trc
        from osim.virtual_markers import extract_ssm67_markers, validate_marker_map

        self.assertEqual(len(MARKER_NAMES), 67)
        self.assertTrue(callable(ensure_ssm67_marker_xml))
        self.assertTrue(callable(write_trc))
        self.assertTrue(callable(extract_ssm67_markers))
        self.assertTrue(callable(validate_marker_map))

    def test_bio_matrix_shape(self):
        import numpy as np

        from common.biomech import BIOMECH_COMPONENT_KEYS
        from osim.physics import bio_matrix_mint

        q = np.random.randn(30, 33).astype(np.float32)
        bio = bio_matrix_mint(q, fps=20.0)
        self.assertEqual(bio.shape, (30, len(BIOMECH_COMPONENT_KEYS)))

    def test_default_model_path(self):
        from osim.coord_map import default_mint_model_path

        path = default_mint_model_path()
        self.assertTrue(path.is_file(), f"Missing bundled model: {path}")


if __name__ == "__main__":
    unittest.main()
