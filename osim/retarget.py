"""HumanML3D joints → MinT OpenSim generalized coordinates (SSM67 / SMPL-H pipeline)."""

from __future__ import annotations

import os
from typing import Literal

import numpy as np

from osim.opensim_ik import run_mint_opensim_pipeline
from osim.retarget_result import RetargetResult
from osim.smplh_fit import fit_smplh_to_hml_joints
from osim.virtual_markers import extract_ssm67_markers

NUM_HML3D_JOINTS = 22

RetargetMethod = Literal["mint", "bootstrap", "auto"]


def retarget_hml_joints_to_q(
    joints: np.ndarray,
    *,
    fps: float = 20.0,
    method: RetargetMethod = "auto",
    use_opensim_ik: bool = True,  # legacy alias for method selection
    smplh_model_dir: str | None = None,
    smplh_iters: int = 120,
    device: str = "cpu",
) -> RetargetResult:
    """Map ``joints`` ``[T, 22, 3]`` → MinT OpenSim ``q`` ``[T, ndof]``.

    Default pipeline (MinT-faithful):
      HML joints → SMPL-H fit → 67 virtual markers → OpenSim IK (Lai model)

    Fallback ``bootstrap`` uses direct HML→Lai heuristic IK (legacy).
    """
    arr = np.asarray(joints, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != NUM_HML3D_JOINTS or arr.shape[2] != 3:
        raise ValueError(f"Expected joints [T, 22, 3], got {arr.shape}")
    t_len = int(arr.shape[0])
    if t_len < 1:
        raise ValueError("Empty joint sequence")

    requested = str(method).strip().lower()
    chosen = requested
    if chosen == "auto":
        env = os.environ.get("MINT_RETARGET_METHOD", "mint").strip().lower()
        chosen = env if env in {"mint", "bootstrap"} else "mint"
        if not use_opensim_ik:
            chosen = "bootstrap"
    allow_fallback = requested == "auto"

    if chosen == "bootstrap":
        from osim.retarget_bootstrap import retarget_bootstrap

        return retarget_bootstrap(arr)

    try:
        fit = fit_smplh_to_hml_joints(
            arr,
            model_dir=smplh_model_dir,
            num_iters=int(smplh_iters),
            device=str(device),
        )
        markers = extract_ssm67_markers(fit.vertices)
        q, ik_err = run_mint_opensim_pipeline(markers, fps=fps)
        if q.shape[0] != t_len:
            q = q[: min(q.shape[0], t_len)]
        return RetargetResult(
            q=q.astype(np.float32),
            mean_fk_error=float(ik_err),
            num_frames=int(q.shape[0]),
            method="mint",
            mean_smpl_joint_error_m=float(fit.mean_joint_error_m),
        )
    except Exception as exc:
        if not allow_fallback:
            raise
        from osim.retarget_bootstrap import retarget_bootstrap

        out = retarget_bootstrap(arr)
        out.method = f"bootstrap_after_error:{type(exc).__name__}"
        return out


def retarget_heuristic(joints: np.ndarray) -> RetargetResult:
    """Legacy alias for bootstrap retargeting."""
    from osim.retarget_bootstrap import retarget_bootstrap

    return retarget_bootstrap(joints)
