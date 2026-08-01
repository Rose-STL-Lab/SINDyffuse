from __future__ import annotations
import hashlib
import os
from pathlib import Path
from typing import Tuple
import torch
from common.distributed import barrier, is_main_process
from common.paths import repo_root
__all__ = ['clip_download_root', 'load_clip']

def clip_download_root() -> Path:
    raw = os.environ.get('CLIP_DOWNLOAD_ROOT', '').strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (repo_root() / '.cache' / 'clip').resolve()

def _clip_weight_path(model_name: str, download_root: Path) -> Path:
    import clip.clip as clip_core
    url = clip_core._MODELS[model_name]
    return download_root / os.path.basename(url)

def _purge_clip_weight(model_name: str, download_root: Path) -> None:
    path = _clip_weight_path(model_name, download_root)
    if path.is_file():
        path.unlink()

def _load_clip_once(model_name: str, *, device: torch.device, jit: bool, download_root: Path) -> Tuple[torch.nn.Module, object]:
    import clip
    return clip.load(model_name, device=device, jit=jit, download_root=str(download_root))

def load_clip(model_name: str, device: torch.device, *, jit: bool=False) -> Tuple[torch.nn.Module, object]:
    import clip
    if model_name not in clip.available_models():
        raise ValueError(f'Unknown CLIP model {model_name!r}; choose from {clip.available_models()}')
    root = clip_download_root()
    root.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        last_err: RuntimeError | None = None
        for attempt in range(3):
            try:
                _load_clip_once(model_name, device=torch.device('cpu'), jit=jit, download_root=root)
                last_err = None
                break
            except RuntimeError as exc:
                if 'SHA256' not in str(exc):
                    raise
                last_err = exc
                _purge_clip_weight(model_name, root)
        if last_err is not None:
            raise last_err
        weight = _clip_weight_path(model_name, root)
        if weight.is_file():
            import clip.clip as clip_core
            expected = clip_core._MODELS[model_name].split('/')[-2]
            digest = hashlib.sha256(weight.read_bytes()).hexdigest()
            if digest != expected:
                weight.unlink()
                raise RuntimeError(f'CLIP weight checksum mismatch after download: {weight}')
    barrier()
    return _load_clip_once(model_name, device=device, jit=jit, download_root=root)