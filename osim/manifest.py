"""JSON manifests, signatures, and model inspection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_contact_model_config(model_path: str) -> Dict[str, Any]:
    p = Path(model_path)
    if not p.exists():
        return {"model_exists": False, "has_contact_components": False, "contact_component_count": 0}
    txt = p.read_text(encoding="utf-8", errors="ignore")
    tags = [
        "SmoothSphereHalfSpaceForce",
        "HuntCrossleyForce",
        "ContactSphere",
        "ContactHalfSpace",
    ]
    count = 0
    for tag in tags:
        count += txt.count(f"<{tag}")
    return {
        "model_exists": True,
        "has_contact_components": bool(count > 0),
        "contact_component_count": int(count),
    }


def write_run_manifest(output_dir: Path, payload: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / "compute_biomechanics_manifest.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def trial_features_signature(mem: Dict[str, Any]) -> Dict[str, Any]:
    """Return JSON-safe signature of an in-memory result (no arrays)."""
    out: Dict[str, Any] = {"source": mem.get("source", None), "trials": []}
    for tr in mem.get("trials", []):
        feats = tr.get("features", {})
        out["trials"].append(
            {
                "trial_idx": int(tr.get("trial_idx", -1)),
                "trial_name": str(tr.get("trial_name", "")),
                "num_frames": int(tr.get("num_frames", 0)),
                "num_dofs": int(tr.get("num_dofs", 0)),
                "dt": float(tr.get("dt", 0.0)),
                "keys": sorted(list(feats.keys())),
                "shapes": {k: list(getattr(feats[k], "shape", [])) for k in feats.keys()},
                "dtypes": {k: str(getattr(feats[k], "dtype", "")) for k in feats.keys()},
            }
        )
    return out
