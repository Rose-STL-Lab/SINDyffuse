from __future__ import annotations
import unittest
import sys
import types
from unittest.mock import MagicMock, patch

sys.modules.setdefault('casadi', types.ModuleType('casadi'))
sys.modules.setdefault('torch', types.ModuleType('torch'))
opensim = sys.modules.setdefault('opensim', types.ModuleType('opensim'))
opensim.Logger = MagicMock()

import nimble.opensimad_track as track


class OpenSimAdIsolationTest(unittest.TestCase):
    def test_single_segment_uses_one_task_spawned_pool(self) -> None:
        result = (0, 'activations', True, {'solver_success': True}, 'grf')
        future = MagicMock()
        future.result.return_value = result
        executor = MagicMock()
        executor.__enter__.return_value.submit.return_value = future
        executor.__exit__.return_value = False
        with patch.object(track, 'ProcessPoolExecutor', return_value=executor) as pool, patch.object(
            track, 'get_context', return_value='spawn-context'
        ):
            actual = track.solve_one_opensimad_segment_isolated(('job',))
        self.assertEqual(actual, result)
        kwargs = pool.call_args.kwargs
        self.assertEqual(kwargs['max_workers'], 1)
        self.assertEqual(kwargs['mp_context'], 'spawn-context')
        if track.sys.version_info >= (3, 11):
            self.assertEqual(kwargs['max_tasks_per_child'], 1)


if __name__ == '__main__':
    unittest.main()