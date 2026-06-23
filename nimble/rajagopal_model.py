"""Shared Rajagopal model preparation for MocoTrack and static optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import opensim as osim

from nimble.muscle_activation import MuscleActivationConfig, rajagopal_model_path

_MOCO_TOE_JOINTS: Tuple[str, ...] = ("mtp_r", "mtp_l")


def unlock_rajagopal_coordinates(model: osim.Model) -> None:
    for i in range(model.getCoordinateSet().getSize()):
        coord = model.getCoordinateSet().get(i)
        if coord.get_locked():
            coord.set_locked(False)


def prepare_unlocked_rajagopal_base(work_dir: Path) -> Path:
    """Write Rajagopal2015 with all coordinates unlocked."""
    out = work_dir / "rajagopal_unlocked.osim"
    if out.is_file():
        return out
    model = osim.Model(str(rajagopal_model_path()))
    unlock_rajagopal_coordinates(model)
    model.initSystem()
    model.printToXML(str(out))
    return out


def build_activation_model_processor(
    base_model_path: Path,
    cfg: MuscleActivationConfig,
    *,
    weld_toe_joints: bool = False,
) -> osim.ModelProcessor:
    """DeGroote muscles + residuals + reserves (shared by Moco and static opt)."""
    mp = osim.ModelProcessor(str(base_model_path))
    if weld_toe_joints and cfg.moco_weld_toe_joints:
        joints_to_weld = osim.StdVectorString()
        for joint_name in _MOCO_TOE_JOINTS:
            joints_to_weld.append(joint_name)
        mp.append(osim.ModOpReplaceJointsWithWelds(joints_to_weld))
    mp.append(osim.ModOpIgnoreTendonCompliance())
    mp.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    mp.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    mp.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
    mp.append(
        osim.ModOpAddResiduals(
            float(cfg.moco_residual_force),
            float(cfg.moco_residual_force) * 0.2,
            1.0,
        )
    )
    reserve_force = float(cfg.moco_reserve_optimal_force) * float(cfg.moco_reserve_scale)
    mp.append(osim.ModOpAddReserves(reserve_force, 1.0))
    return mp


def muscle_names_from_processor(mp: osim.ModelProcessor) -> Tuple[str, ...]:
    model = mp.process()
    model.initSystem()
    muscles = model.getMuscles()
    return tuple(muscles.get(i).getName() for i in range(muscles.getSize()))


def prepare_rajagopal_activation_model(
    work_dir: Path,
    cfg: MuscleActivationConfig,
    *,
    weld_toe_joints: bool = False,
) -> Tuple[Path, Tuple[str, ...]]:
    """Processed Rajagopal model path and ordered muscle names."""
    work_dir.mkdir(parents=True, exist_ok=True)
    tag = "moco" if weld_toe_joints else "static"
    out = work_dir / f"rajagopal_activation_{tag}.osim"
    base = prepare_unlocked_rajagopal_base(work_dir)
    mp = build_activation_model_processor(base, cfg, weld_toe_joints=weld_toe_joints)
    processed = mp.process()
    processed.initSystem()
    muscles = processed.getMuscles()
    names = tuple(muscles.get(i).getName() for i in range(muscles.getSize()))
    if not out.is_file():
        processed.printToXML(str(out))
    return out, names
