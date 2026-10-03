from __future__ import annotations
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import yaml
sys.modules.setdefault('torch', types.ModuleType('torch'))
from nimble.opensimad import polynomial_shards as shards

ROOT = Path(__file__).resolve().parent.parent

class PolynomialShardsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'build'
        self.root.mkdir()
        self.values = np.arange(23 * 2, dtype=np.float64).reshape(23, 2)
        (self.root / f'{shards.MODEL_NAME}.osim').write_text('test model')
        (self.root / 'DummyMotionFullRange.mot').write_text('test motion')
        shards.atomic_npz(self.root / 'samples.npz', coordinate_values=self.values)
        self.manifest = {
            'schema_version': 1, 'chunk_frames': 10, 'sample_count': 23, 'chunk_count': 3,
            'muscle_names': ['test_r'], 'coordinate_labels': ['/jointset/a/value', '/jointset/b/value'],
            'model_sha256': shards.file_hash(self.root / f'{shards.MODEL_NAME}.osim'),
            'motion_sha256': shards.file_hash(self.root / 'DummyMotionFullRange.mot'),
            'samples_sha256': shards.file_hash(self.root / 'samples.npz'),
            'code': shards.code_identity(), 'runtime': {'test': True},
        }
        self.manifest['build_id'] = shards.build_identity(self.manifest)
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))
        self.runtime = patch.object(shards, 'runtime_identity', return_value={'test': True})
        self.runtime.start()
        self.addCleanup(self.runtime.stop)
        self.vendor = types.ModuleType('muscleDataOpenSimAD')
        self.calls = []
        def extract(args):
            _, rows, _, index = args
            self.calls.append(index)
            return rows[:, :1].copy(), rows[:, None, :].copy()
        self.vendor._get_mtu_chunk_isolated = MagicMock(side_effect=extract)
        # Avoid importing the OpenSim-dependent preparation module for unit tests.
        prep = types.ModuleType('nimble.opensimad.polynomial_cache')
        prep._ensure_vendor_on_path = lambda: None
        prep.RIGHT_JOINTS = ('a',); prep.LEFT_JOINTS = ('b',)
        prep.RIGHT_MUSCLES = ('test_r',); prep.LEFT_MUSCLES = ('test_l',)
        modules = patch.dict(sys.modules, {'muscleDataOpenSimAD': self.vendor, 'nimble.opensimad.polynomial_cache': prep})
        modules.start()
        self.addCleanup(modules.stop)

    def extract(self, index):
        return shards.extract_shard(self.root, index=index, work_dir=Path(self.temporary.name) / 'scratch', expected_chunks=3)

    def test_out_of_order_completion_assembles_exact_original_order(self) -> None:
        for index in (2, 0, 1):
            self.extract(index)
        manifest, fitting = shards.assemble_shards(self.root, expected_chunks=3)
        self.assertEqual(manifest['build_id'], self.manifest['build_id'])
        np.testing.assert_array_equal(fitting['coordinate_values'], self.values)
        np.testing.assert_array_equal(fitting['mtu_lengths'], self.values[:, :1])
        np.testing.assert_array_equal(fitting['mtu_moment_arms'], self.values[:, None, :])
        self.assertEqual(fitting['coordinate_names'], ['a', 'b'])
        self.assertEqual([shards.chunk_range(manifest, i) for i in range(3)], [(0,10),(10,20),(20,23)])

    def test_retry_reuses_complete_chunk_and_recomputes_corrupt_chunk(self) -> None:
        path = self.extract(1)
        before = path.read_bytes()
        self.extract(1)
        self.assertEqual(self.calls, [1])
        self.assertEqual(before, path.read_bytes())
        path.write_bytes(b'partial write')
        self.extract(1)
        self.assertEqual(self.calls, [1, 1])
        shards.validate_chunk(path, self.manifest, self.values, 1)

    def test_failed_extraction_never_publishes_result(self) -> None:
        self.vendor._get_mtu_chunk_isolated.side_effect = RuntimeError('worker died')
        with self.assertRaisesRegex(RuntimeError, 'worker died'):
            self.extract(0)
        self.assertFalse(shards.chunk_path(self.root, 0).exists())
        self.vendor._get_mtu_chunk_isolated.side_effect = None
        self.vendor._get_mtu_chunk_isolated.return_value = (np.ones((10, 1)), np.full((10, 1, 2), np.nan))
        with self.assertRaisesRegex(ValueError, 'Invalid result values'):
            self.extract(0)
        self.assertFalse(shards.chunk_path(self.root, 0).exists())

    def test_missing_and_stale_results_block_finalization(self) -> None:
        self.extract(0)
        with self.assertRaises(FileNotFoundError):
            shards.assemble_shards(self.root)
        for index in (1, 2): self.extract(index)
        path = shards.chunk_path(self.root, 1)
        with np.load(path) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        metadata = json.loads(arrays['metadata'].item()); metadata['build_id'] = 'stale'
        arrays['metadata'] = np.array(json.dumps(metadata)); shards.atomic_npz(path, **arrays)
        with self.assertRaisesRegex(ValueError, 'Stale'):
            shards.assemble_shards(self.root)

    def test_input_drift_and_task_mismatch_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, 'task count'):
            shards.load_inputs(self.root, expected_chunks=4)
        with self.assertRaises(ValueError): self.extract(3)
        with patch.object(shards, 'runtime_identity', return_value={'changed': True}):
            with self.assertRaisesRegex(ValueError, 'runtime changed'):
                shards.load_inputs(self.root)
        (self.root / f'{shards.MODEL_NAME}.osim').write_text('modified')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            shards.load_inputs(self.root)

    def test_finalizer_fits_both_sides_from_complete_dataset(self) -> None:
        for index in range(3): self.extract(index)
        self.vendor.getMTParameters = MagicMock()
        seen = []
        def fit(load, folder, **kwargs):
            self.assertFalse(load)
            dataset = np.load(Path(folder) / f'data4PolynomialFitting_{shards.MODEL_NAME}_default.npy', allow_pickle=True).item()
            np.testing.assert_array_equal(dataset['mtu_lengths'], self.values[:, :1])
            seen.append(kwargs['side'])
            self.assertNotIn('overwritedata4PolynomialFitting', kwargs)
        self.vendor.getPolynomialData = MagicMock(side_effect=fit)
        published = self.root / 'published.json'
        published.write_text(json.dumps({'artifacts': {'a': 'hash'}}))
        with patch('nimble.opensimad.paths.ad_scaled_adjusted_model_path', return_value=self.root / f'{shards.MODEL_NAME}.osim'), patch(
            'nimble.opensimad.paths.publish_polynomial_cache', return_value=published
        ) as publish, patch('nimble.opensimad.paths.opensimad_dir', return_value=self.root), patch(
            'nimble.opensimad.paths.validate_polynomial_cache', return_value={'artifacts': {'a': 'hash'}}
        ):
            result = shards.finalize_shards(self.root, work_dir=Path(self.temporary.name) / 'scratch')
        self.assertEqual(seen, ['r', 'l'])
        publish.assert_called_once()
        self.assertEqual(result['build_id'], self.manifest['build_id'])
        self.assertEqual(json.loads(published.read_text())['distributed_build_id'], self.manifest['build_id'])

    def test_deployment_configuration_is_consistent(self) -> None:
        base = ROOT / 'deploy/jobs/preprocess-dataset'
        jobs = [yaml.safe_load((base / name / 'job.yaml').read_text()) for name in (
            'prepare-opensimad-polynomials', 'build-opensimad-polynomials', 'finalize-opensimad-polynomials')]
        worker = jobs[1]['spec']
        self.assertEqual(worker['parallelism'], 100)
        self.assertEqual(worker['completions'], 200)
        self.assertEqual(worker['completionMode'], 'Indexed')
        configs = [{x['name']: x['value'] for x in job['spec']['template']['spec']['containers'][0]['env']} for job in jobs]
        self.assertEqual(len({c['POLYNOMIAL_BUILD_DIR'] for c in configs}), 1)
        self.assertTrue(all(int(c['POLYNOMIAL_EXPECTED_CHUNKS']) == worker['completions'] for c in configs))

    def test_preparation_is_immutable_and_reuses_identical_snapshot(self) -> None:
        target = Path(self.temporary.name) / 'prepared'
        source = self.root / f'{shards.MODEL_NAME}.osim'
        dummy = self.root / 'DummyMotionFullRange.mot'
        osim = MagicMock()
        osim.TimeSeriesTable.return_value.getColumnLabels.return_value = self.manifest['coordinate_labels']
        osim.TimeSeriesTable.return_value.getMatrix.return_value.to_numpy.return_value = self.values
        force = osim.Model.return_value.getForceSet.return_value
        force.getSize.return_value = 1
        force.get.return_value.getConcreteClassName.return_value = 'Millard2012EquilibriumMuscle'
        force.get.return_value.getName.return_value = 'test_r'
        prep = sys.modules['nimble.opensimad.polynomial_cache']
        def write_dummy(src, dest):
            dest.write_bytes(src.read_bytes())
            return dest
        model_prep = types.ModuleType('nimble.opensimad.model_prep')
        model_prep.ensure_ad_ready_artifacts = MagicMock()
        with patch.object(prep, '_write_full_range_dummy', side_effect=write_dummy, create=True), patch.dict(
            sys.modules, {'opensim': osim, 'nimble.opensimad.model_prep': model_prep}
        ), patch('nimble.opensimad.paths.ad_scaled_adjusted_model_path', return_value=source), patch(
            'nimble.opensimad.paths.vendor_dummy_motion_path', return_value=dummy
        ), patch('nimble.opensimad.paths.validate_compiled_external'):
            first = shards.prepare_shards(target, chunk_frames=10, expected_chunks=3)
            (target / 'chunks').mkdir()
            sentinel = target / 'chunks' / 'sentinel'
            sentinel.write_text('keep')
            second = shards.prepare_shards(target, chunk_frames=10, expected_chunks=3)
            self.assertEqual(first, second)
            self.assertEqual(sentinel.read_text(), 'keep')
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                shards.prepare_shards(target, chunk_frames=5)
            with self.assertRaisesRegex(ValueError, 'Expected 4 tasks'):
                shards.prepare_shards(target, chunk_frames=10, expected_chunks=4)

    def test_modified_finite_result_fails_checksum(self) -> None:
        path = self.extract(0)
        with np.load(path) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays['mtu_lengths'][0, 0] += 1
        shards.atomic_npz(path, **arrays)
        with self.assertRaisesRegex(ValueError, 'Stale or misindexed'):
            shards.validate_chunk(path, self.manifest, self.values, 0)

if __name__ == '__main__':
    unittest.main()