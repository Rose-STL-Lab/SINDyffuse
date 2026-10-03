from __future__ import annotations
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
# Follow the existing OpenSim tests: no Torch execution is needed here.
sys.modules.setdefault('torch', types.ModuleType('torch'))
from common.memory_diagnostics import DIAGNOSTICS_ENV, diagnostic_event, diagnostic_stage, memory_monitor, memory_snapshot

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / 'nimble' / 'opensimad' / 'vendor' / 'opencap_ad'

def load_vendor(name: str):
    spec = importlib.util.spec_from_file_location(f'test_{name}', VENDOR / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class PolynomialDiagnosticsTest(unittest.TestCase):
    def test_entrypoint_records_failure_before_dependency_imports(self) -> None:
        spec = importlib.util.spec_from_file_location('polynomial_entrypoint', ROOT / 'scripts' / 'build_lai_opensimad_polynomials.py')
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {}, clear=False), patch.object(
            sys, 'argv', ['build_lai_opensimad_polynomials.py', '--log_dir', root]
        ), patch.object(script, '_build', side_effect=RuntimeError('native dependency failure')):
            with self.assertRaisesRegex(RuntimeError, 'native dependency failure'):
                script.main()
            paths = list(Path(root).glob('*.jsonl'))
            self.assertEqual(len(paths), 1)
            records = [json.loads(line) for line in paths[0].read_text().splitlines()]
            self.assertEqual(records[0]['event'], 'run_start')
            self.assertEqual(records[-1]['event'], 'stage_failed')
            self.assertIn('native dependency failure', records[-1]['traceback'])

    def test_linux_memory_units_and_cgroup_counters(self) -> None:
        files = {'/proc/self/status': 'VmRSS: 123 kB\n',
                 '/sys/fs/cgroup/memory.current': '1000',
                 '/sys/fs/cgroup/memory.peak': '2000',
                 '/sys/fs/cgroup/memory.max': 'max',
                 '/sys/fs/cgroup/memory.events': 'oom 2\noom_kill 1\n'}
        def read(path):
            if str(path) not in files:
                raise FileNotFoundError(str(path))
            return files[str(path)]
        with patch('pathlib.Path.read_text', read), patch('common.memory_diagnostics.sys.platform', 'linux'), patch(
            'common.memory_diagnostics.resource.getrusage', return_value=types.SimpleNamespace(ru_maxrss=456)
        ):
            snapshot = memory_snapshot()
        self.assertEqual(snapshot['rss_current_bytes'], 123 * 1024)
        self.assertEqual(snapshot['rss_peak_bytes'], 456 * 1024)
        self.assertEqual(snapshot['memory.events']['oom_kill'], 1)
        self.assertEqual(snapshot['memory.max'], 'max')

    def test_chunk_isolation_keeps_one_fresh_spawned_worker(self) -> None:
        module = load_vendor('muscleDataOpenSimAD')
        pool = MagicMock()
        pool.__enter__.return_value.submit.return_value.result.return_value = 'result'
        with patch.object(module, 'ProcessPoolExecutor', return_value=pool) as executor, patch.object(
            module, 'get_context', return_value='spawn-context'
        ) as context:
            self.assertEqual(module._get_mtu_chunk_isolated(('job',)), 'result')
        context.assert_called_once_with('spawn')
        executor.assert_called_once_with(max_workers=1, mp_context='spawn-context')

    def test_events_survive_worker_exit_and_record_failure_and_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'events.jsonl'
            with patch.dict(os.environ, {DIAGNOSTICS_ENV: str(path)}):
                diagnostic_event('parent')
                subprocess.run([sys.executable, '-c',
                                'import sys, types; sys.modules["torch"] = types.ModuleType("torch"); '
                                'from common.memory_diagnostics import diagnostic_event; diagnostic_event("child")'],
                               cwd=ROOT, check=True)
                with memory_monitor(0.01):
                    time.sleep(0.05)
                with self.assertRaisesRegex(RuntimeError, 'test failure'):
                    with diagnostic_stage('fitting', side='r'):
                        raise RuntimeError('test failure')
            records = [json.loads(line) for line in path.read_text().splitlines()]
            events = [record['event'] for record in records]
            self.assertIn('child', events)
            self.assertIn('memory_heartbeat', events)
            failure = next(record for record in records if record['event'] == 'stage_failed')
            self.assertIn('test failure', failure['traceback'])
            self.assertEqual(failure['side'], 'r')
            self.assertTrue(all('rss_peak_bytes' in record and 'timestamp' in record for record in records))
            self.assertNotEqual(records[0]['pid'], records[1]['pid'])

    def test_disabled_logging_does_not_create_files(self) -> None:
        with patch.dict(os.environ, {DIAGNOSTICS_ENV: ''}), patch('pathlib.Path.open') as opened:
            diagnostic_event('disabled')
            opened.assert_not_called()

    def test_fresh_build_retains_all_samples_and_reuses_saved_data(self) -> None:
        module = load_vendor('muscleDataOpenSimAD')
        data = np.arange(23 * 2, dtype=float).reshape(23, 2)
        osim = MagicMock()
        osim.TimeSeriesTable.return_value.getColumnLabels.return_value = ['/jointset/a/value', '/jointset/b/value']
        osim.TimeSeriesTable.return_value.getMatrix.return_value.to_numpy.return_value = data
        forces = osim.Model.return_value.getForceSet.return_value
        forces.getSize.return_value = 1
        forces.get.return_value.getConcreteClassName.return_value = 'Millard2012EquilibriumMuscle'
        forces.get.return_value.getName.return_value = 'test_r'
        fitter = MagicMock()
        fitter.getPolynomialCoefficients.return_value = {'unchanged': True}
        chunks = []
        def extract(args):
            _, rows, _, index = args
            chunks.append((index, rows.copy()))
            return rows[:, :1].copy(), rows[:, None, :].copy()
        with tempfile.TemporaryDirectory() as root, patch.dict(
            sys.modules, {'opensim': osim, 'polynomialsOpenSimAD': fitter}
        ), patch.object(module, '_get_mtu_chunk_isolated', side_effect=extract):
            kwargs = dict(pathModelFolder=root, modelName='test', joints=['a', 'b'], muscles=['test_r'], chunk_frames=10)
            result = module.getPolynomialData(False, **kwargs)
            self.assertEqual(result, {'unchanged': True})
            self.assertEqual([len(rows) for _, rows in chunks], [10, 10, 3])
            np.testing.assert_array_equal(np.concatenate([rows for _, rows in chunks]), data)
            fresh = fitter.getPolynomialCoefficients.call_args.args[0]
            np.testing.assert_array_equal(fresh['coordinate_values'], data)
            np.testing.assert_array_equal(fresh['mtu_lengths'], data[:, :1])
            module.getPolynomialData(False, **kwargs)
            self.assertEqual(len(chunks), 3)
            cached = fitter.getPolynomialCoefficients.call_args.args[0]
            np.testing.assert_array_equal(cached['mtu_moment_arms'], fresh['mtu_moment_arms'])
            self.assertEqual(fitter.getPolynomialCoefficients.call_args.kwargs, {'side': ''})

    def test_polynomial_logging_preserves_coefficients(self) -> None:
        module = load_vendor('polynomialsOpenSimAD')
        x = np.linspace(-0.5, 0.5, 50)
        dataset = {'coordinate_names': ['a'], 'coordinate_values': np.rad2deg(x[:, None]),
                   'muscle_names': ['test_r'], 'mtu_lengths': (0.4 + 0.1 * x + 0.02 * x**2)[:, None],
                   'mtu_moment_arms': (-0.1 - 0.04 * x)[:, None, None]}
        kwargs = dict(joints=['a'], muscles=['test_r'], removeBadHipFlexionEntries=False)
        with patch.dict(os.environ, {DIAGNOSTICS_ENV: ''}):
            baseline = module.getPolynomialCoefficients(dataset, **kwargs)
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {DIAGNOSTICS_ENV: str(Path(root) / 'fit.jsonl')}):
            logged = module.getPolynomialCoefficients(dataset, **kwargs)
        np.testing.assert_array_equal(logged['test_r']['coefficients'], baseline['test_r']['coefficients'])
        self.assertEqual(logged['test_r']['order'], 3)
        np.testing.assert_allclose(logged['test_r']['coefficients'], [0.4, 0.1, 0.02, 0], atol=1e-12)

    def test_generalized_basis_matches_legacy_order_and_derivatives(self) -> None:
        module = load_vendor('polynomialsOpenSimAD')
        rng = np.random.default_rng(3)
        for dimension in range(1, 6):
            x = rng.uniform(-0.5, 0.5, (8, dimension))
            basis = module.polynomial_estimation(dimension, 3)
            powers = module._polynomial_exponents(dimension, 3)
            np.testing.assert_array_equal(basis.getVariables(x), np.column_stack([
                module._monomial(x.T, p) * np.ones(len(x)) for p in powers]))
            for component in range(dimension):
                np.testing.assert_array_equal(basis.getVariableDerivatives(x, component), np.column_stack([
                    module._monomial(x.T, p, component) * np.ones(len(x)) for p in powers]))

    def test_six_coordinate_fit_and_runtime_evaluation(self) -> None:
        module = load_vendor('polynomialsOpenSimAD')
        x = np.random.default_rng(9).uniform(-0.5, 0.5, (200, 6))
        slopes = np.arange(1, 7) * 0.01
        dataset = {'coordinate_names': [f'q{i}' for i in range(6)],
                   'coordinate_values': np.rad2deg(x), 'muscle_names': ['gasmed_r'],
                   'mtu_lengths': (0.4 + x @ slopes)[:, None],
                   'mtu_moment_arms': np.broadcast_to(-slopes, (len(x), 1, 6)).copy()}
        result = module.getPolynomialCoefficients(dataset, dataset['coordinate_names'], ['gasmed_r'],
                                                  removeBadHipFlexionEntries=False)['gasmed_r']
        self.assertEqual(result['dimension'], 6)
        self.assertEqual(result['order'], 3)
        self.assertEqual(len(result['coefficients']), 84)
        polynomial = module.polynomials(result['coefficients'], 6, 3)
        for row in x[:5]:
            self.assertAlmostEqual(polynomial.calcValue(row), 0.4 + row @ slopes, places=12)
            for component in range(6):
                self.assertAlmostEqual(polynomial.calcDerivative(row, component), slopes[component], places=12)

if __name__ == '__main__':
    unittest.main()