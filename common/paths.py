from __future__ import annotations
import os
from pathlib import Path
NIMBLE_B3D_SUBDIR = 'nimble_b3d'
__all__ = ['NIMBLE_B3D_SUBDIR', 'cleanup_preprocess_manifests', 'default_datasets_dir', 'default_humanml3d_root', 'humanml3d_text_dir', 'nimble_b3d_dir', 'repo_root', 'resolve_data_root', 'resolve_repo_path', 'results_dir', 'sindy_latest_link', 'activation_surrogate_latest_link', 'diffusion_latest_link', 'update_latest_symlink']

def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent

def default_datasets_dir() -> Path:
    return repo_root() / 'datasets'

def default_humanml3d_root() -> str:
    explicit = os.environ.get('HUMANML3D_ROOT', '').strip()
    if explicit:
        return explicit
    return str(default_datasets_dir() / 'HumanML3D')

def nimble_b3d_dir(data_root: str | Path, *, subdir: str | None=None) -> Path:
    name = str(subdir).strip() if subdir else NIMBLE_B3D_SUBDIR
    return Path(data_root).expanduser().resolve() / name

def humanml3d_text_dir(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser().resolve()
    texts_subdir = root / 'texts'
    if texts_subdir.is_dir():
        return texts_subdir
    return root

def resolve_data_root(path: str | None) -> str:
    if path is None or str(path).strip() == '':
        return default_humanml3d_root()
    return str(resolve_repo_path(path))

def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (repo_root() / p).resolve()

def results_dir() -> Path:
    return repo_root() / 'results'

def sindy_latest_link() -> Path:
    return results_dir() / 'sindy' / 'latest'

def activation_surrogate_latest_link() -> Path:
    return results_dir() / 'activation_surrogate' / 'latest'

def diffusion_latest_link(guidance: str) -> Path:
    return results_dir() / 'diffusion' / str(guidance).strip().lower() / 'latest'

def update_latest_symlink(*, run_dir: Path, latest_link: Path) -> None:
    run = run_dir.resolve()
    latest_link.parent.mkdir(parents=True, exist_ok=True)
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run, target_is_directory=True)

def cleanup_preprocess_manifests(out_root: str | Path) -> list[str]:
    root = Path(out_root).expanduser().resolve()
    removed: list[str] = []
    for path in sorted(root.glob('preprocess_*_manifest*.jsonl')):
        path.unlink()
        removed.append(str(path))
    meta = root / 'preprocess_meta.json'
    if meta.is_file():
        meta.unlink()
        removed.append(str(meta))
    return removed