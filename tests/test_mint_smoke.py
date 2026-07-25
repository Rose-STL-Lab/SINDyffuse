"""Smoke tests for MinT pipeline modules (no dataset required)."""

from __future__ import annotations

import os
import unittest

import numpy as np


class MintSmokeTests(unittest.TestCase):
    def test_skeleton_config_mint_dims(self) -> None:
        from nimble.channels import BIOMECH_COMPONENT_KEYS

        os.environ["SINDYFFUSE_SKELETON"] = "mint"
        from importlib import reload

        import common.skeleton_config as sc

        reload(sc)
        self.assertEqual(sc.MINT_MUSCLE_COUNT, 402)
        self.assertEqual(sc.n_bio_targets(), len(BIOMECH_COMPONENT_KEYS))
        self.assertEqual(sc.n_muscle_targets(), 402)
        self.assertEqual(sc.n_sindy_targets(), len(BIOMECH_COMPONENT_KEYS) + 402)

    def test_cache_schema_roundtrip(self) -> None:
        from mint.cache_schema import write_motion_cache, read_motion_cache, KEY_Q, KEY_MUSCLE
        import tempfile

        t, ndof, nm = 8, 10, 402
        q = np.random.randn(t, ndof).astype(np.float32)
        act = np.random.rand(t, nm).astype(np.float32)
        with tempfile.TemporaryDirectory() as td:
            path = f"{td}/000001.npz"
            write_motion_cache(path, q=q, muscle_activations=act)
            data = read_motion_cache(path)
            self.assertEqual(data[KEY_Q].shape, (t, ndof))
            self.assertEqual(data[KEY_MUSCLE].shape, (t, nm))

    def test_retarget_bootstrap(self) -> None:
        from mint.retarget import retarget_hml_joints_to_q

        joints = np.random.randn(12, 22, 3).astype(np.float32)
        rt = retarget_hml_joints_to_q(joints, method="bootstrap")
        self.assertEqual(rt.q.shape[0], 12)
        self.assertGreater(rt.q.shape[1], 0)
        self.assertTrue(rt.method.startswith("bootstrap"))

    def test_ssm67_marker_extract_and_trc(self) -> None:
        from mint.ssm67_markers import MARKER_NAMES, ensure_ssm67_marker_xml
        from mint.trc_io import write_trc
        from mint.virtual_markers import extract_ssm67_markers, validate_marker_map
        import tempfile

        validate_marker_map()
        xml_path = ensure_ssm67_marker_xml()
        self.assertTrue(xml_path.is_file())

        verts = np.random.randn(5, 6890, 3).astype(np.float32)
        markers = extract_ssm67_markers(verts)
        self.assertEqual(markers.shape, (5, 67, 3))

        with tempfile.TemporaryDirectory() as td:
            trc = f"{td}/m.trc"
            write_trc(trc, markers, fps=20.0)
            with open(trc, encoding="utf-8") as f:
                text = f.read()
            self.assertIn(MARKER_NAMES[0], text)
            self.assertIn("NumMarkers", text)

    def test_bio_matrix_mint(self) -> None:
        from mint.physics import bio_matrix_mint
        from nimble.channels import BIOMECH_COMPONENT_KEYS

        q = np.random.randn(16, 12).astype(np.float32)
        bio = bio_matrix_mint(q, fps=20.0)
        self.assertEqual(bio.shape, (16, len(BIOMECH_COMPONENT_KEYS)))

    def test_bundled_lai_model_path(self) -> None:
        from mint.coord_map import default_mint_model_path

        p = default_mint_model_path()
        self.assertTrue(p.is_file(), f"Missing bundled model: {p}")


if __name__ == "__main__":
    unittest.main()
