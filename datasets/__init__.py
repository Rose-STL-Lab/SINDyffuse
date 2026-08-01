from __future__ import annotations
from typing import TYPE_CHECKING
from common.paths import NIMBLE_B3D_SUBDIR, default_datasets_dir, default_humanml3d_root, nimble_b3d_dir, repo_root, resolve_data_root
__all__ = ['NimbleDataset', 'compute_nimble_normalization_stats', 'NIMBLE_B3D_SUBDIR', 'default_datasets_dir', 'default_humanml3d_root', 'nimble_b3d_dir', 'repo_root', 'resolve_data_root']

def __getattr__(name: str):
    if name == 'NimbleDataset':
        from .nimble_dataset import NimbleDataset
        return NimbleDataset
    if name == 'compute_nimble_normalization_stats':
        from .nimble_dataset import compute_nimble_normalization_stats
        return compute_nimble_normalization_stats
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
if TYPE_CHECKING:
    from .nimble_dataset import NimbleDataset, compute_nimble_normalization_stats