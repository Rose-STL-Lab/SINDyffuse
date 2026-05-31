from __future__ import annotations

from datasets.nimble_dataset import NimbleDataset

from diffusion.config import DatasetName


def get_dataset(dataset: str, **kwargs):
    name = DatasetName(str(dataset))
    if name == DatasetName.NIMBLE:
        return NimbleDataset(**kwargs)
    raise NotImplementedError(
        f"Dataset {dataset!r} is not supported. Use dataset='nimble' with a B3D cache root."
    )
