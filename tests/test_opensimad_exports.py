from __future__ import annotations
import ast
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / 'nimble/opensimad/vendor/opencap_ad'

class OpenSimAdExportsTest(unittest.TestCase):
    def test_actual_export_block_writes_headless_activation_and_resultant_files(self):
        tree = ast.parse((VENDOR / 'mainOpenSimAD.py').read_text())
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_tracking')
        block = next(node for node in ast.walk(run) if isinstance(node, ast.If)
                     and ast.unparse(node.test) == 'writeGUI or writeMachineReadable')
        module = types.ModuleType('utils')
        writer_tree = ast.parse((VENDOR / 'utils.py').read_text())
        writer = next(node for node in writer_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'numpy_to_storage')
        module.__dict__['np'] = np
        exec(compile(ast.Module(body=[writer], type_ignores=[]), '<storage-writer>', 'exec'), module.__dict__)
        labels = {'GRF': {'all': {}}, 'COP': {'all': {}}, 'GRM': {'all': {}}}
        for side in ('left', 'right'):
            labels['GRF'][side] = {}; labels['COP'][side] = {}; labels['GRM'][side] = {}
            for group in ('GRF', 'COP', 'GRM'):
                cols = [f'{group}_{side}_{i}' for i in range(3)]
                labels[group]['all'][side] = cols
                labels[group][side]['sphere'] = cols
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, {'utils': module}):
            import os
            zeros = {side: np.zeros((3, 2)) for side in ('left', 'right')}
            ns = dict(np=np, os=os, writeGUI=False, writeMachineReadable=True,
                      torque_driven_model=False, joints=['knee_angle_l'], bothSidesMuscles=['muscle_l'],
                      tgridf=np.array([[0., .05, .1]]), Qs_opt_nsc_deg=np.array([[10.,20.,30.]]),
                      a_opt=np.array([[.1,.2,.3]]), pathResults=root, trialName='segment', case='0',
                      torques_opt=np.zeros((1,2)), nContactSpheres=2, contactSides=['left','right'],
                      contactSpheres={'left':['sphere'],'right':['sphere']}, GR_labels=labels,
                      GRF_s_opt={side:{'sphere':zeros[side]} for side in zeros},
                      COP_s_opt={side:{'sphere':zeros[side]} for side in zeros},
                      GRF_all_opt=zeros, COP_all_opt=zeros, freeT_all_opt=zeros)
            code = compile(ast.Module(body=[block], type_ignores=[]), '<actual-vendor-export>', 'exec')
            exec(code, ns)
            act = Path(root) / 'kinematics_activations_segment_0.mot'
            resultant = Path(root) / 'GRF_resultant_segment_0.mot'
            self.assertTrue(act.is_file() and resultant.is_file())
            self.assertIn('inDegrees=yes', act.read_text())
            self.assertIn('muscle_l/activation', act.read_text())
            self.assertIn('inDegrees=no', resultant.read_text())
            for path in Path(root).glob('*.mot'): path.unlink()
            ns['writeMachineReadable'] = False
            exec(code, ns)
            self.assertFalse(list(Path(root).glob('*.mot')))

    def test_wrapper_requests_headless_exports_and_resultant_grf(self):
        tree = ast.parse((ROOT / 'nimble/opensimad/track_segment.py').read_text())
        call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == 'run_tracking')
        options = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        self.assertFalse(options['writeGUI'])
        self.assertTrue(options['writeMachineReadable'])
        source = (ROOT / 'nimble/opensimad/track_segment.py').read_text()
        self.assertEqual(source.count("'GRF_resultant_*.mot'"), 2)