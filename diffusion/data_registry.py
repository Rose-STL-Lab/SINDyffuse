from __future__ import annotations

from diffusion.config import DatasetName
from diffusion.text_dataset import HumanML3DTextMotionDataset


def get_text_motion_dataset(dataset: str, **kwargs):
    name = DatasetName(str(dataset))
    if name == DatasetName.HUMANML3D:
        return HumanML3DTextMotionDataset(**kwargs)
    raise NotImplementedError(f"Dataset not supported yet: {dataset}")

