from __future__ import annotations
import sys
import types
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.modules.setdefault('torch', types.ModuleType('torch'))
if not hasattr(sys.modules['torch'], 'Tensor'):
    sys.modules['torch'].Tensor = type('Tensor', (), {})
from nimble.opensimad.canary_selection import filtered_bounds_violations
from nimble.opensimad.mint_settings import MINT_POLYNOMIAL_BOUNDS

NAMES = tuple(MINT_POLYNOMIAL_BOUNDS)

class CanarySelectionTest(unittest.TestCase):
    def test_in_domain_motion_is_unchanged(self):
        q = np.deg2rad(np.array([(b['min'] + b['max']) / 2 for b in MINT_POLYNOMIAL_BOUNDS.values()]))
        q = np.tile(q, (34, 1))
        before = q.copy()
        self.assertEqual(filtered_bounds_violations(q, NAMES, fps=20), {})
        np.testing.assert_array_equal(q, before)

    def test_filter_overshoot_is_rejected_without_clipping(self):
        q = np.zeros((34, len(NAMES)))
        col = NAMES.index('hip_adduction_r')
        q[12:24, col] = np.deg2rad(30)
        result = filtered_bounds_violations(q, NAMES, fps=20)
        self.assertGreater(result['hip_adduction_r']['max_degrees'], 30)
        self.assertAlmostEqual(float(np.max(np.rad2deg(q[:, col]))), 30)

    def test_invalid_data_and_timing_rejected(self):
        for fps in (0, float('nan')):
            with self.assertRaises(ValueError):
                filtered_bounds_violations(np.zeros((34, len(NAMES))), NAMES, fps=fps)

    def test_selector_skips_high_variability_out_of_domain_and_explicit_id_fails(self):
        sys.modules.setdefault('casadi', types.ModuleType('casadi'))
        opensim = sys.modules.setdefault('opensim', types.ModuleType('opensim'))
        if not hasattr(opensim, 'Logger'):
            from unittest.mock import MagicMock
            opensim.Logger = MagicMock()
        from scripts import run_opensimad_canary as script
        from nimble.lai_coord_map import LAI_CACHE_DOF_NAMES
        good = np.zeros((34, len(LAI_CACHE_DOF_NAMES)))
        bad = good.copy()
        bad[:, LAI_CACHE_DOF_NAMES.index('hip_rotation_r')] = np.deg2rad(np.linspace(-60, 60, 34))
        with tempfile.TemporaryDirectory() as root:
            cache = Path(root)
            for name in ('bad', 'good'): (cache / f'{name}.npz').touch()
            def read(path, **kwargs):
                return {'q': bad if path.stem == 'bad' else good, 'fps': 20}
            with patch.object(script, 'read_motion_npz', side_effect=read):
                selected, _ = script._select_motion(cache, '', 34)
                self.assertEqual(selected.stem, 'good')
                with self.assertRaisesRegex(ValueError, 'Explicit canary motion'):
                    script._select_motion(cache, 'bad', 34)

if __name__ == '__main__':
    unittest.main()