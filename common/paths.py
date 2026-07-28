"""Central, clone-friendly path defaults for SINDyffuse.

Layout (default)::

    <repo>/datasets/HumanML3D/          # dataset root (texts, splits, raw motion)
        new_joints/ | new_joint_vecs/
        Mean.npy, Std.npy               # HumanML3D 263-D normalization (source only)
        mint_cache/                   # MinT OpenSim q + 402 muscle activations (npz)
            {motion_id}.npz
            Mean.npy, Std.npy
"""

from __future__ import annotations

import os
from pathlib import Path

# Subdirectory under the HumanML3D root for per-motion MinT NPZ exports.
MINT_CACHE_SUBDIR = "mint_cache"

__all__ = [
    "MINT_CACHE_SUBDIR",
    "baselines_dir",
    "baseline_output_dir",
    "cleanup_preprocess_manifests",
    "default_datasets_dir",
    "default_humanml3d_root",
    "default_mint_root",
    "humanml3d_text_dir",
    "mint_cache_dir",
    "motion_cache_dir",
    "repo_root",
    "resolve_data_root",
    "resolve_repo_path",
    "results_dir",
    "sindy_latest_link",
    "activation_surrogate_latest_link",
    "diffusion_latest_link",
    "update_latest_symlink",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_datasets_dir() -> Path:
    return repo_root() / "datasets"


def default_humanml3d_root() -> str:
    explicit = os.environ.get("HUMANML3D_ROOT", "").strip()
    if explicit:
        return explicit
    return str(default_datasets_dir() / "HumanML3D")


def mint_cache_dir(data_root: str | Path, *, subdir: str | None = None) -> Path:
    """Path to per-motion MinT NPZ cache (default ``mint_cache/`` under the dataset root)."""
    name = str(subdir).strip() if subdir else MINT_CACHE_SUBDIR
    return Path(data_root).expanduser().resolve() / name


def default_mint_root() -> str:
    explicit = os.environ.get("MINT_ROOT", "").strip()
    if explicit:
        return explicit
    return str(default_datasets_dir() / "MinT")


def motion_cache_dir(
    data_root: str | Path,
    *,
    subdir: str | None = None,
) -> Path:
    """Resolve motion cache directory."""
    name = str(subdir).strip() if subdir else MINT_CACHE_SUBDIR
    return mint_cache_dir(data_root, subdir=name)


def humanml3d_text_dir(data_root: str | Path) -> Path:
    """Directory with per-motion caption files (``{motion_id}.txt``).

    HumanML3D releases usually use ``<root>/texts/``; some checkouts keep the same
    ``*.txt`` files directly under the dataset root instead.
    """
    root = Path(data_root).expanduser().resolve()
    texts_subdir = root / "texts"
    if texts_subdir.is_dir():
        return texts_subdir
    return root


def resolve_data_root(path: str | None) -> str:
    if path is None or str(path).strip() == "":
        return default_humanml3d_root()
    return str(resolve_repo_path(path))


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a config path relative to the repo root when not absolute."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (repo_root() / p).resolve()


def results_dir() -> Path:
    return repo_root() / "results"


def baselines_dir() -> Path:
    """Root for cloned baseline repos and their pre-trained checkpoints."""
    explicit = os.environ.get("SINDYFFUSE_BASELINES_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return repo_root() / "baselines"


def baseline_output_dir(repo_name: str) -> Path:
    """Directory where a baseline method writes generated motions for eval."""
    return baselines_dir() / repo_name / "outputs"


def sindy_latest_link() -> Path:
    return results_dir() / "sindy" / "latest"


def activation_surrogate_latest_link() -> Path:
    return results_dir() / "activation_surrogate" / "latest"


def diffusion_latest_link(guidance: str) -> Path:
    return results_dir() / "diffusion" / str(guidance).strip().lower() / "latest"


def update_latest_symlink(*, run_dir: Path, latest_link: Path) -> None:
    """Point ``latest_link`` at ``run_dir`` (overwrites prior symlink)."""
    run = run_dir.resolve()
    latest_link.parent.mkdir(parents=True, exist_ok=True)
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run, target_is_directory=True)


def cleanup_preprocess_manifests(out_root: str | Path) -> list[str]:
    """Remove temporary preprocess manifest files from the dataset root."""
    root = Path(out_root).expanduser().resolve()
    removed: list[str] = []
    for path in sorted(root.glob("preprocess_manifest*.jsonl")):
        path.unlink()
        removed.append(str(path))
    meta = root / "preprocess_meta.json"
    if meta.is_file():
        meta.unlink()
        removed.append(str(meta))
    return removed
