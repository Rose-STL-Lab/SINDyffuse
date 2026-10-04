"""Executed in a fresh subprocess to avoid other tests' Torch/OpenSim stubs."""
import sys
import types
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.modules.setdefault('opensim', types.ModuleType('opensim'))
from surrogate.dataset import ActivationB3DDataset, collate_activation_windows
from surrogate.model import build_activation_surrogate
from surrogate.losses import activation_surrogate_loss

class VariableWindowsTest(unittest.TestCase):
    def dataset(self, root, q, act, mask, **kwargs):
        root = Path(root); (root / 'partial.npz').touch()
        data = {'q': q, 'muscle_activations': act, 'muscle_activation_mask': mask, 'num_frames': len(q)}
        with patch('surrogate.dataset.lai_cache_dir', return_value=root), patch('surrogate.dataset.load_split_ids', return_value=['partial']), patch('surrogate.dataset.read_motion_npz', return_value=data):
            ds = ActivationB3DDataset(str(root), normalize_q=False, **kwargs)
        ds._cache[str(root / 'partial.npz')] = data
        return ds

    def test_partial_motion_and_single_frame_are_retained(self):
        with tempfile.TemporaryDirectory() as root:
            q = np.ones((40,31), np.float32); act = np.full((40,80), np.nan, np.float32); mask = np.zeros(40)
            act[:28] = .2; mask[:28] = 1; act[35] = .3; mask[35] = 1
            ds = self.dataset(root, q, act, mask)
            self.assertEqual(len(ds), 2)
            self.assertEqual([ds[i][0].shape[0] for i in range(2)], [28,1])
            self.assertEqual(ds.num_motions_kept, 1)
            self.assertEqual(ds.num_valid_frames, 29)
            qbatch, abatch, valid = next(iter(DataLoader(ds, batch_size=2, collate_fn=collate_activation_windows)))
            self.assertEqual(tuple(qbatch.shape), (2,28,31))
            self.assertEqual(valid.sum().item(),29)
            for kind in ('mlp','transformer'):
                model = build_activation_surrogate(model_type=kind, hidden_dim=16, num_layers=1, num_heads=2, dim_feedforward=16, dropout=0, max_seq_len=64)
                pred = model(qbatch, valid_mask=valid)
                loss = activation_surrogate_loss(pred, abatch, valid_mask=valid)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_padding_and_gaps_do_not_contribute_to_loss(self):
        pred = torch.tensor([[[.2],[.4],[99.]]],requires_grad=True)
        target = torch.tensor([[[.1],[.2],[float('nan')]]])
        mask = torch.tensor([[True,True,False]])
        loss = activation_surrogate_loss(pred,target,valid_mask=mask,lambda_temporal=.5)
        self.assertAlmostEqual(loss.item(),.2,places=6)
        loss.backward(); self.assertEqual(pred.grad[0,2,0].item(),0)
        one = activation_surrogate_loss(pred[:,:1],target[:,:1],valid_mask=mask[:,:1])
        self.assertAlmostEqual(one.item(),.1,places=6)

    def test_tail_coverage_and_configurable_minimum(self):
        with tempfile.TemporaryDirectory() as root:
            q=np.ones((101,31),np.float32); a=np.ones((101,80),np.float32)*.3; m=np.ones(101)
            ds=self.dataset(root,q,a,m,window_size=28,window_stride=16,min_window_size=2)
            covered=set()
            for _,start,n in ds._windows: covered.update(range(start,start+n))
            self.assertEqual(covered,set(range(101)))
            with self.assertRaises(ValueError): self.dataset(root,q[:1],a[:1],m[:1],min_window_size=2)

    def test_nonfinite_pose_breaks_runs_and_relaxed_flags_do_not_allow_nan_labels(self):
        with tempfile.TemporaryDirectory() as root:
            q=np.ones((10,31),np.float32); a=np.ones((10,80),np.float32)*.3; m=np.ones(10)
            q[3]=np.nan; a[7]=np.nan
            ds=self.dataset(root,q,a,m,skip_invalid_activations=False,min_valid_fraction=0)
            self.assertEqual([n for _,_,n in ds._windows],[3,3,2])
            for i in range(len(ds)):
                inp,target=ds[i]
                self.assertTrue(torch.isfinite(inp).all() and torch.isfinite(target).all())

    def test_transformer_valid_predictions_ignore_padding(self):
        model=build_activation_surrogate(model_type='transformer',num_layers=1,num_heads=2,dim_feedforward=16,dropout=0,max_seq_len=64).eval()
        q=torch.randn(1,5,31); mask=torch.tensor([[True,True,False,False,False]])
        altered=q.clone(); altered[:,2:]=100
        with torch.no_grad():
            torch.testing.assert_close(model(q,mask)[:,:2],model(altered,mask)[:,:2])

if __name__ == '__main__':
    unittest.main()