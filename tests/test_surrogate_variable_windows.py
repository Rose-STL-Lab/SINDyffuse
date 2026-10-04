import subprocess
import sys
import unittest
from pathlib import Path

class SurrogateVariableWindowsTest(unittest.TestCase):
    def test_pilot_uses_production_shard_mapping(self):
        import yaml
        root=Path(__file__).resolve().parent.parent
        rendered=subprocess.check_output(['kubectl','kustomize',str(root/'deploy/jobs/preprocess-dataset/moco-track-pilot')],text=True)
        job=yaml.safe_load(rendered)
        self.assertEqual(job['spec']['parallelism'],5)
        self.assertEqual(job['spec']['completions'],5)
        env={entry['name']:entry['value'] for entry in job['spec']['template']['spec']['containers'][0]['env']}
        self.assertEqual(env['PREPROCESS_NUM_SHARDS'],'180')
        self.assertEqual(env['MAX_MOTIONS'],'900')

    def test_real_torch_variable_window_pipeline(self):
        # Look up Torch outside this process, where legacy tests install stubs.
        probe = subprocess.run([sys.executable, '-c', 'import torch; assert hasattr(torch, "Tensor")'], capture_output=True)
        if probe.returncode:
            self.skipTest('Real Torch is unavailable in this interpreter')
        path = Path(__file__).with_name('surrogate_variable_windows_check.py')
        result = subprocess.run([sys.executable, str(path)], cwd=path.parent.parent,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)