from __future__ import annotations
import unittest
import sys
import types
opensim = sys.modules.setdefault('opensim', types.ModuleType('opensim'))
opensim.Logger = type('Logger', (), {})
from types import SimpleNamespace
import numpy as np
from nimble.opensimad.label_processing import stitch_solve_windows, tracking_metrics, pool_tracking, set_grf_validity

def spec(index, start, end, core_start, core_end):
    return SimpleNamespace(index=index,solve_start=start,solve_end=end,core_start=core_start,
                           core_end=core_end,solve_len=end-start,core_len=core_end-core_start)

class LabelProcessingTest(unittest.TestCase):
    def test_crossfade_uses_buffers_and_reduces_boundary_jump(self):
        specs=[spec(0,0,7,0,5),spec(1,3,10,5,10)]
        values=[np.zeros((7,2)),np.ones((7,2))]
        out=stitch_solve_windows(10,specs,values,[True,True],blend_frames=2)
        np.testing.assert_allclose(out[3:7,0],[.2,.4,.6,.8])
        self.assertLess(abs(out[5,0]-out[4,0]),1)
        np.testing.assert_array_equal(out[:3],0)
        np.testing.assert_array_equal(out[7:],1)

    def test_failed_core_stays_nan_despite_successful_buffers(self):
        specs=[spec(0,0,7,0,5),spec(1,3,12,5,10),spec(2,8,15,10,15)]
        out=stitch_solve_windows(15,specs,[np.zeros((7,1)),np.ones((9,1)),np.ones((7,1))],
                                 [True,False,True],blend_frames=2)
        self.assertTrue(np.isnan(out[5:10]).all())
        np.testing.assert_array_equal(out[:5],0)
        np.testing.assert_array_equal(out[10:],1)

    def test_one_frame_tail_and_disabled_blending(self):
        specs=[spec(0,0,6,0,5),spec(1,3,6,5,6)]
        out=stitch_solve_windows(6,specs,[np.zeros((6,1)),np.ones((3,1))],[True,True],blend_frames=3)
        self.assertTrue(np.isfinite(out).all())
        direct=stitch_solve_windows(6,specs,[np.zeros((6,1)),np.ones((3,1))],[True,True],blend_frames=0)
        self.assertEqual(direct[-1,0],1)

    def test_tracking_units_and_analysis_flags_not_gates(self):
        names=['knee_angle_r','pelvis_tx']
        ref=np.zeros((10,2)); sim=np.tile([np.deg2rad(6),.03],(10,1))
        m=tracking_metrics(sim,ref,names)
        self.assertAlmostEqual(m['max_rotational_rmse_deg'],6)
        self.assertAlmostEqual(m['max_translational_rmse_m'],.03)
        self.assertFalse(m['mint_analysis_flags']['rotational_rmse_lt_5deg'])
        self.assertFalse(m['mint_analysis_flags']['translational_rmse_lt_0_02m'])
        lower=tracking_metrics(np.tile([np.deg2rad(2),.01],(30,1)),np.zeros((30,2)),names)
        pooled=pool_tracking([m,lower])
        self.assertAlmostEqual(pooled['max_rotational_rmse_deg'],np.sqrt((36*10+4*30)/40))
        self.assertEqual(pooled['per_coordinate'][0]['sample_count'],40)
        self.assertIsNone(pool_tracking([])['max_rotational_rmse_deg'])

    def test_grf_trust_requires_finite_force_and_moment_channels(self):
        g=np.ones((3,18)); g[-1,:12]=np.nan; g[1,3]=np.nan
        out=set_grf_validity(g)
        np.testing.assert_array_equal(out[:,17],[1,0,0])
        self.assertTrue(np.isnan(out[-1,:17]).all())

    def test_diagnostics_survive_npz_round_trip(self):
        import tempfile, json
        from pathlib import Path
        from datasets.lai_cache import write_motion_npz, read_motion_npz
        diagnostics={'label_processing_version':'test','coordinate_tracking':tracking_metrics(
            np.zeros((2,2)),np.zeros((2,2)),['knee_angle_r','pelvis_tx'])}
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'motion.npz'
            write_motion_npz(path,motion_id='motion',q=np.ones((2,31)),
                             extra={'activation_diagnostics_json':np.asarray(json.dumps(diagnostics))})
            self.assertEqual(read_motion_npz(path)['activation_diagnostics'],diagnostics)