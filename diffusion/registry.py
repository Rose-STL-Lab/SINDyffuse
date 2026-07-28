from __future__ import annotations

from datasets.mint_motion_dataset import MintMotionDataset

from diffusion.config import DatasetName


def get_dataset(dataset: str, **kwargs):
    name = DatasetName(str(dataset))
    if name == DatasetName.MINT:
        return MintMotionDataset(**kwargs)
    raise NotImplementedError(
        f"Dataset {dataset!r} is not supported. Use dataset='mint'."
    )
