from __future__ import annotations
from datasets.nimble_dataset import NimbleDataset
from diffusion.config import DatasetName

def get_dataset(dataset: str, **kwargs):
    name = str(dataset).strip().lower()
    if name in {DatasetName.NIMBLE.value, DatasetName.LAI.value, 'nimble', 'lai'}:
        return NimbleDataset(**kwargs)
    raise NotImplementedError(f"Dataset {dataset!r} is not supported. Use dataset='lai' with a lai_cache NPZ root.")
