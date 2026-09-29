from __future__ import annotations
import unittest

from nimble.opensimad.mint_settings import mint_tracking_settings


class MintTrackingSettingsTest(unittest.TestCase):
    def test_workers_require_compiled_and_precomputed_artifacts(self) -> None:
        settings = mint_tracking_settings()
        self.assertFalse(settings['useExpressionGraphFunction'])
        self.assertTrue(settings['requirePrecomputedPolynomials'])
        self.assertEqual(settings['meshDensity'], 50)
        self.assertEqual(settings['max_iterations'], 2500)
        self.assertEqual(settings['precomputedPolynomialBounds']['knee_angle_r']['max'], 140)


if __name__ == '__main__':
    unittest.main()