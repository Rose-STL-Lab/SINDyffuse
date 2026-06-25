"""Pre-activation quality metrics (IK, clip length).

Gating is no longer enforced before muscle activation; motions always attempt
``moco_track`` / ``static_optimization`` and repair missing activations later.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from nimble.muscle_activation import MuscleActivationConfig
from nimble.rajagopal_coord_map import RAJAGOPAL_NIMBLE_DOF_NAMES


def _ik_mean_fk_loss(ik_stats: Dict[str, Any]) -> float | None:
    val = ik_stats.get("mean_fk_loss")
    if val is None:
        val = ik_stats.get("mean_fit_joints_loss", ik_stats.get("mean_ik_error"))
    if val is None:
        return None
    out = float(val)
    return out if np.isfinite(out) else None


def ik_success_fraction(ik_stats: Dict[str, Any]) -> float:
    ratio = ik_stats.get("success_ratio")
    if ratio is not None:
        return float(ratio)
    total = float(ik_stats.get("total_frames", 0))
    success = float(ik_stats.get("success_count", 0))
    return float(success / max(total, 1.0))


def evaluate_activation_gate(
    ik_stats: Dict[str, Any],
    *,
    num_frames: int,
    cfg: MuscleActivationConfig,
) -> Tuple[bool, str]:
    """Advisory gate metrics only (not enforced during export)."""
    if int(num_frames) < int(cfg.moco_min_frames):
        return False, f"frame_count {num_frames} < min_frames {cfg.moco_min_frames}"

    min_ik = float(cfg.moco_min_ik_success_fraction)
    if min_ik > 0.0:
        frac = ik_success_fraction(ik_stats)
        if frac < min_ik:
            return False, f"ik_success_fraction {frac:.4f} < {min_ik:.4f}"

    if cfg.moco_max_mean_fk_loss is not None and float(cfg.moco_max_mean_fk_loss) > 0.0:
        loss = _ik_mean_fk_loss(ik_stats)
        if loss is not None and loss > float(cfg.moco_max_mean_fk_loss):
            return False, f"mean_fk_loss {loss:.6g} > {cfg.moco_max_mean_fk_loss:.6g}"

    if cfg.moco_max_pelvis_ty_range_m > 0.0:
        pelvis_range = float(ik_stats.get("pelvis_ty_range_m", np.nan))
        if np.isfinite(pelvis_range) and pelvis_range > float(cfg.moco_max_pelvis_ty_range_m):
            return (
                False,
                f"pelvis_ty_range_m {pelvis_range:.4f} > "
                f"{cfg.moco_max_pelvis_ty_range_m:.4f}",
            )

    return True, ""


def pelvis_ty_range_m(q: np.ndarray) -> float:
    """Vertical pelvis excursion in meters (Nimble ``ground_pelvis_4`` = ty).

    Expects ``q`` as ``[T, ndof]`` (same layout as MocoTrack input). If passed
    ``[ndof, T]`` (Nimble B3D pose layout), it is transposed automatically.
    """
    arr = np.asarray(q, dtype=np.float64)
    if arr.ndim != 2:
        return float("nan")
    ndof = len(RAJAGOPAL_NIMBLE_DOF_NAMES)
    if arr.shape[1] == ndof:
        pass
    elif arr.shape[0] == ndof:
        arr = arr.T
    else:
        return float("nan")
    if arr.shape[1] < 5:
        return float("nan")
    ty = arr[:, 4]
    finite = np.isfinite(ty)
    if finite.sum() < 2:
        return float("nan")
    return float(ty[finite].max() - ty[finite].min())


def summarize_moco_metadata(metadata: Dict[str, Any]) -> Dict[str, float | str | int]:
    """Flatten Moco run metadata for preprocess manifest / logs."""
    out: Dict[str, float | str | int] = {
        "repaired_frame_count": int(metadata.get("repaired_frame_count", 0)),
    }
    obj = metadata.get("moco_objective")
    if obj is not None and np.isfinite(float(obj)):
        out["moco_objective"] = float(obj)

    solve_details = metadata.get("moco_solve_details")
    if isinstance(solve_details, dict):
        if solve_details.get("solver_status"):
            out["moco_solver_status"] = str(solve_details["solver_status"])
        if solve_details.get("solver_success") is not None:
            out["moco_solver_success"] = int(bool(solve_details["solver_success"]))
    elif metadata.get("moco_solver_status"):
        out["moco_solver_status"] = str(metadata["moco_solver_status"])
    if metadata.get("moco_solver_success") is not None and "moco_solver_success" not in out:
        out["moco_solver_success"] = int(bool(metadata["moco_solver_success"]))

    # Legacy segmented metadata (older preprocess runs).
    details = metadata.get("moco_segment_details") or []
    if details and "moco_solver_status" not in out:
        statuses = [
            str(d.get("solver_status", ""))
            for d in details
            if isinstance(d, dict) and d.get("solver_status")
        ]
        if statuses:
            out["moco_solver_status"] = statuses[-1]
    if details and "moco_solver_success" not in out:
        solver_ok = any(
            isinstance(d, dict) and d.get("solver_success") for d in details
        )
        out["moco_solver_success"] = int(solver_ok)

    return out
