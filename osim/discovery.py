"""Phase-0 utilities: inspect MinT layout, HML overlap, and bundled models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.paths import default_humanml3d_root, default_mint_root, repo_root
from datasets.splits import load_split_ids


def inspect_mint_dataset_root(mint_root: str | Path) -> Dict[str, Any]:
    """Summarize MinT tarball layout (kinematics pickles, metadata, sample counts)."""
    root = Path(mint_root).expanduser().resolve()
    out: Dict[str, Any] = {"mint_root": str(root), "exists": root.is_dir()}
    if not root.is_dir():
        return out

    patterns = {
        "muscle_pkl": list(root.glob("**/muscle_activations*.pkl"))[:5],
        "metadata_pkl": list(root.glob("**/metadata*.pkl"))[:5],
        "mot_files": list(root.glob("**/*.mot"))[:5],
        "coord_pkl": list(root.glob("**/*coord*.pkl"))[:5],
    }
    out["sample_paths"] = {k: [str(p.relative_to(root)) for p in v] for k, v in patterns.items()}
    out["counts"] = {k: len(list(root.glob(g)) if g.startswith("**") else root.glob(g)) for k, g in {
        "muscle_pkl_total": "**/muscle_activations*.pkl",
        "mot_total": "**/*.mot",
    }.items()}

    try:
        from musint.datasets.mint_dataset import MintDataset

        ds = MintDataset(str(root), use_cache=True, load_humanml3d_names=True)
        out["mint_num_samples"] = len(ds)
    except Exception as exc:
        out["mint_dataset_error"] = str(exc)

    return out


def humanml3d_mint_overlap(
    *,
    data_root: Optional[str] = None,
    mint_root: Optional[str] = None,
    split: str = "train",
) -> Dict[str, Any]:
    """Count HumanML3D motions with MinT label coverage."""
    hml_root = Path(data_root or default_humanml3d_root()).expanduser().resolve()
    mint_path = Path(mint_root or default_mint_root()).expanduser().resolve()
    ids = load_split_ids(str(hml_root), split)
    covered: List[str] = []
    missing: List[str] = []
    errors: Dict[str, str] = {}

    try:
        from musint.datasets.mint_dataset import MintDataset

        ds = MintDataset(str(mint_path), use_cache=True, load_humanml3d_names=True)
        for sid in ids:
            try:
                ds.by_humanml3d_name(str(sid))
                covered.append(sid)
            except ValueError:
                missing.append(sid)
            except Exception as exc:
                errors[sid] = str(exc)
    except ImportError as exc:
        return {
            "error": f"musint not installed: {exc}",
            "split": split,
            "num_hml_ids": len(ids),
        }

    return {
        "split": split,
        "hml_root": str(hml_root),
        "mint_root": str(mint_path),
        "num_hml_ids": len(ids),
        "num_mint_covered": len(covered),
        "num_mint_missing": len(missing),
        "coverage_fraction": len(covered) / max(len(ids), 1),
        "sample_covered": covered[:10],
        "sample_missing": missing[:10],
        "errors": dict(list(errors.items())[:5]),
    }


def bundled_model_status() -> Dict[str, Any]:
    from osim.coord_map import default_mint_model_path

    status: Dict[str, Any] = {"models_dir": str(repo_root() / "models" / "mint")}
    try:
        p = default_mint_model_path()
        status["lai_model"] = str(p)
        status["lai_model_bytes"] = p.stat().st_size
    except FileNotFoundError as exc:
        status["lai_model_error"] = str(exc)

    bruno = repo_root() / "models" / "mint" / "BrunoThoracolumbar.osim"
    status["bruno_model"] = str(bruno) if bruno.is_file() else None
    status["bruno_note"] = (
        "Bruno thoracolumbar model must be obtained from the MinT release / paper supplement."
    )
    return status


def run_discovery(
    *,
    mint_root: Optional[str] = None,
    data_root: Optional[str] = None,
) -> Dict[str, Any]:
    report = {
        "bundled_models": bundled_model_status(),
        "overlap_train": humanml3d_mint_overlap(data_root=data_root, mint_root=mint_root, split="train"),
        "overlap_val": humanml3d_mint_overlap(data_root=data_root, mint_root=mint_root, split="val"),
    }
    mr = mint_root or default_mint_root()
    if Path(mr).expanduser().is_dir():
        report["mint_layout"] = inspect_mint_dataset_root(mr)
    else:
        report["mint_layout"] = {"mint_root": mr, "exists": False}
    return report


def write_discovery_report(path: str | Path, **kwargs: Any) -> Path:
    report = run_discovery(**kwargs)
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out
