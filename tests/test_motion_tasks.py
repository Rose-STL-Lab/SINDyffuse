from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
import types
import numpy as np
sys.modules.setdefault('torch', types.ModuleType('torch'))
sys.modules.setdefault('casadi', types.ModuleType('casadi'))
opensim = sys.modules.setdefault('opensim', types.ModuleType('opensim'))
if not hasattr(opensim, 'Logger'):
    from unittest.mock import MagicMock
    opensim.Logger = MagicMock()
from common.motion_tasks import prepare_tasks, load_tasks, task_motion, load_outcomes, atomic_json, outcome_path
from datasets.lai_cache import write_motion_npz, read_motion_npz

class MotionTasksTest(unittest.TestCase):
    def test_manifest_is_immutable_and_assigns_exactly_one_motion(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / 'tasks'
            kwargs = dict(motion_ids=['a','b','c'], configuration={'segments':8},artifacts={'model':'hash'},expected_tasks=3,data_root=Path(root))
            tasks = prepare_tasks(directory,**kwargs)
            self.assertEqual([task_motion(tasks,i,3) for i in range(3)],['a','b','c'])
            self.assertEqual(prepare_tasks(directory,**kwargs),tasks)
            with self.assertRaises(ValueError): task_motion(tasks,0,2)
            with self.assertRaises(ValueError): task_motion(tasks,3,3)
            kwargs['motion_ids']=['b','a','c']
            with self.assertRaisesRegex(ValueError,'changed'): prepare_tasks(directory,**kwargs)
            self.assertEqual(load_tasks(directory/'tasks.json'),tasks)

    def test_all_outcomes_required_and_identity_validated(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root)
            tasks=prepare_tasks(directory,motion_ids=['a','b'],configuration={},artifacts={},expected_tasks=2,data_root=directory)
            row={'id':'a','task_index':0,'task_set_id':tasks['task_set_id'],'status':'moco_failed'}
            atomic_json(outcome_path(directory,0),row)
            with self.assertRaises(FileNotFoundError): load_outcomes(directory)
            row.update(id='b',task_index=1,status='ok')
            atomic_json(outcome_path(directory,1),row)
            self.assertEqual(len(load_outcomes(directory)[1]),2)
            row['task_set_id']='stale'; atomic_json(outcome_path(directory,1),row)
            with self.assertRaisesRegex(ValueError,'identity'): load_outcomes(directory)

    def test_atomic_npz_preserves_old_file_on_write_failure(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'motion.npz'
            write_motion_npz(path,motion_id='motion',q=np.ones((4,31)))
            before=path.read_bytes()
            def fail(fp,**payload):
                fp.write(b'partial archive')
                raise RuntimeError('compression failure')
            with patch('datasets.lai_cache.np.savez_compressed',side_effect=fail):
                with self.assertRaises(RuntimeError): write_motion_npz(path,motion_id='motion',q=np.zeros((4,31)))
            self.assertEqual(path.read_bytes(),before)
            self.assertFalse(list(Path(root).glob('.*')))
            np.testing.assert_array_equal(read_motion_npz(path)['q'],np.ones((4,31)))

    def test_worker_terminal_failure_is_not_retried_and_success_is_reused(self):
        from scripts import preprocess_moco as worker
        from common.run_logging import null_logger
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); directory=root/'tasks'
            args=SimpleNamespace(out_root=str(root),hml_root=str(root),motion_task_dir=str(directory),motion_task_mode='process',
                fps=20,mass_kg=70,expected_tasks=1,motion_task_index=0,skip_existing=True)
            cfg={}
            tasks=prepare_tasks(directory,motion_ids=['a'],configuration=cfg,artifacts={'test':True,'execution':{'solver_threads':1,'segment_workers':1}},expected_tasks=1,data_root=root)
            with patch.object(worker,'_task_artifacts',return_value={'test':True}), patch.object(worker,'muscle_activation_config_from_args',return_value=cfg), patch.object(worker,'muscle_activation_config_to_dict',return_value=cfg), patch.object(worker,'_process_one_moco',return_value={'id':'a','status':'moco_failed','meta':{'moco_failed_reason':'OpenSimAD optimization did not converge: Maximum_Iterations_Exceeded'}}) as process:
                worker.run_motion_task(args,null_logger())
                worker.run_motion_task(args,null_logger())
                process.assert_called_once()
            self.assertEqual(json.loads(outcome_path(directory,0).read_text())['status'],'moco_failed')

    def test_worker_unknown_failure_is_retryable(self):
        from scripts import preprocess_moco as worker
        from common.run_logging import null_logger
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); directory=root/'tasks'
            args=SimpleNamespace(out_root=str(root),hml_root=str(root),motion_task_dir=str(directory),motion_task_mode='process',
                fps=20,mass_kg=70,expected_tasks=1,motion_task_index=0,skip_existing=True)
            prepare_tasks(directory,motion_ids=['a'],configuration={},artifacts={'execution':{'solver_threads':1,'segment_workers':1}},expected_tasks=1,data_root=root)
            with patch.object(worker,'_task_artifacts',return_value={}), patch.object(worker,'muscle_activation_config_from_args',return_value={}), patch.object(worker,'muscle_activation_config_to_dict',return_value={}), patch.object(worker,'_process_one_moco',return_value={'id':'a','status':'moco_failed','meta':{'moco_failed_reason':'missing export'}}) as process:
                for _ in range(2):
                    with self.assertRaisesRegex(RuntimeError,'missing export'): worker.run_motion_task(args,null_logger())
                self.assertEqual(process.call_count,2)

    def test_existing_labels_are_imported_once_then_checksum_verified(self):
        from scripts import preprocess_moco as worker
        from common.run_logging import null_logger
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); directory=root/'tasks'; cache=root/'lai_cache'; cache.mkdir()
            output=cache/'a.npz'
            write_motion_npz(output,motion_id='a',q=np.ones((4,31)),
                             muscle_activations=np.full((4,80),.2),muscle_activation_mask=np.ones(4))
            prepare_tasks(directory,motion_ids=['a'],configuration={},artifacts={'execution':{'solver_threads':1,'segment_workers':1}},expected_tasks=1,data_root=root)
            args=SimpleNamespace(out_root=str(root),hml_root=str(root),motion_task_dir=str(directory),motion_task_mode='process',
                fps=20,mass_kg=70,expected_tasks=1,motion_task_index=0,skip_existing=True)
            with patch.object(worker,'_task_artifacts',return_value={}), patch.object(worker,'muscle_activation_config_from_args',return_value={}), patch.object(worker,'muscle_activation_config_to_dict',return_value={}), patch.object(worker,'_process_one_moco',return_value={'id':'a','status':'skipped','path':str(output)}) as process:
                worker.run_motion_task(args,null_logger())
                worker.run_motion_task(args,null_logger())
                process.assert_called_once()
            row=json.loads(outcome_path(directory,0).read_text())
            self.assertEqual(row['valid_frames'],4)
            self.assertTrue(row['existing_labels_imported'])
            self.assertEqual(len(row['result_sha256']),64)

    def test_task_normalization_requires_all_terminal_outcomes(self):
        import ast
        from types import SimpleNamespace
        root_repo=Path(__file__).resolve().parent.parent
        tree=ast.parse((root_repo/'scripts/compute_normalization.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_normalization')
        from common.run_logging import null_logger
        ns={'Path':Path,'json':json,'null_logger':null_logger,'default_humanml3d_root':lambda:'.',
            'NIMBLE_B3D_SUBDIR':'lai_cache','nimble_b3d_dir':lambda root:root/'lai_cache',
            'compute_nimble_normalization_stats':lambda root:{'mean_path':'mean'},
            '_shard_manifest_path':lambda root,i,stage:root/f'legacy.{i}.jsonl'}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'<normalization>', 'exec'),ns)
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); directory=root/'tasks'
            tasks=prepare_tasks(directory,motion_ids=['a','b'],configuration={},artifacts={},expected_tasks=2,data_root=root)
            args=SimpleNamespace(out_root=str(root),motion_task_dir=str(directory),num_shards=180)
            with self.assertRaises(FileNotFoundError): ns['compute_normalization'](args)
            for i,mid in enumerate(['a','b']):
                atomic_json(outcome_path(directory,i),{'id':mid,'task_index':i,'task_set_id':tasks['task_set_id'],
                                                       'status':'ok' if i==0 else 'moco_failed','num_dofs':31})
            meta=ns['compute_normalization'](args)
            self.assertEqual(meta['motions_ok'],1)
            self.assertEqual(meta['motions_error'],1)
            self.assertEqual(meta['motion_task_count'],2)
            self.assertTrue((root/'preprocess_manifest.jsonl').exists())
