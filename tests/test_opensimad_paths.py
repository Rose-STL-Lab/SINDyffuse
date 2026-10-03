from __future__ import annotations
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.modules.setdefault('torch', types.ModuleType('torch'))
from nimble.opensimad import paths


class OpenSimAdArtifactTest(unittest.TestCase):
    def _write_polynomials(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        names = paths.required_polynomial_cache_names()
        np.save(root / names[0], np.ones((5, 2)))
        np.save(root / names[1], np.ones((5, 2)))
        np.save(root / names[2], {'muscle_l': {'coefficients': np.array([0.1])}})
        np.save(root / names[3], {'muscle_r': {'coefficients': np.array([0.1])}})
        np.save(root / names[4], {
            'mtu_lengths': np.ones((2, 2)),
            'mtu_moment_arms': np.ones((2, 2, 2)),
            'muscle_names': ['a', 'b'],
            'coordinate_names': ['x', 'y'],
            'coordinate_values': np.ones((2, 2)),
        })

    def test_publish_polynomial_cache_writes_completion_marker_last(self) -> None:
        with tempfile.TemporaryDirectory() as source_s, tempfile.TemporaryDirectory() as dest_s:
            source = Path(source_s)
            dest = Path(dest_s)
            model = dest / 'model.osim'
            model.write_text('model', encoding='utf-8')
            self._write_polynomials(source)
            with patch.object(paths, 'opensimad_dir', return_value=dest), patch.object(
                paths, 'ad_scaled_adjusted_model_path', return_value=model
            ):
                metadata_path = paths.publish_polynomial_cache(source)
                metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
                self.assertTrue(metadata['complete'])
                validated = paths.validate_polynomial_cache(require_metadata=True, deep=True)
                self.assertEqual(validated['artifacts'], metadata['artifacts'])

    def test_missing_compiled_external_never_falls_back_to_python(self) -> None:
        with tempfile.TemporaryDirectory() as root_s:
            root = Path(root_s)
            with patch.object(paths, 'external_function_dir', return_value=root):
                with self.assertRaisesRegex(FileNotFoundError, 'Missing compiled'):
                    paths.validate_compiled_external(load_library=False, require_metadata=False)

    def test_old_pipeline_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as source_s, tempfile.TemporaryDirectory() as dest_s:
            source, dest = Path(source_s), Path(dest_s)
            model = dest / 'model.osim'; model.write_text('model')
            self._write_polynomials(source)
            with patch.object(paths, 'opensimad_dir', return_value=dest), patch.object(paths, 'ad_scaled_adjusted_model_path', return_value=model):
                metadata_path = paths.publish_polynomial_cache(source)
                metadata = json.loads(metadata_path.read_text())
                metadata.pop('pipeline_version')
                metadata_path.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(RuntimeError, 'pipeline_version'):
                    paths.validate_polynomial_cache()


if __name__ == '__main__':
    unittest.main()