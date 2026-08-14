from __future__ import annotations
from pathlib import Path
from typing import Tuple
import opensim as osim
from common.paths import repo_root
from nimble.muscle_activation import MuscleActivationConfig, rajagopal_model_path
_MOCO_TOE_JOINTS: Tuple[str, ...] = ('mtp_r', 'mtp_l')

def unlock_rajagopal_coordinates(model: osim.Model) -> None:
    for i in range(model.getCoordinateSet().getSize()):
        coord = model.getCoordinateSet().get(i)
        if coord.get_locked():
            coord.set_locked(False)

def prepare_unlocked_rajagopal_base(work_dir: Path) -> Path:
    out = work_dir / 'rajagopal_unlocked.osim'
    if out.is_file():
        return out
    model = osim.Model(str(rajagopal_model_path()))
    unlock_rajagopal_coordinates(model)
    model.initSystem()
    model.printToXML(str(out))
    return out

def prepare_welded_unlocked_rajagopal_base(work_dir: Path) -> Path:
    """Unlocked coords + MTP welds for path-fit / OpenSimAD (no locked joints)."""
    out = Path(work_dir) / 'rajagopal_unlocked_mtp_welded.osim'
    if out.is_file():
        return out
    unlocked = prepare_unlocked_rajagopal_base(work_dir)
    joints = osim.StdVectorString()
    for joint_name in _MOCO_TOE_JOINTS:
        joints.append(joint_name)
    mp = osim.ModelProcessor(str(unlocked))
    mp.append(osim.ModOpReplaceJointsWithWelds(joints))
    model = mp.process()
    model.initSystem()
    model.printToXML(str(out))
    return out

def function_based_path_set_path() -> Path:
    return repo_root() / 'models' / 'rajagopal' / 'Rajagopal2015_FunctionBasedPathSet.xml'

def function_based_path_set_depends_on_mtp(path_set: Path) -> bool:
    """True when the fitted path set still references MTP coordinates.

    Path-fit historically used an unwelded Rajagopal model, so FunctionBasedPath
    expressions depend on mtp_angle_*. MocoTrack welds MTP joints by default,
    which removes those coordinates and makes ModelProcessor::process() fail.
    """
    try:
        text = path_set.read_text(encoding='utf-8', errors='ignore')
    except OSError:
        return False
    return 'mtp_angle' in text

def build_activation_model_processor(base_model_path: Path, cfg: MuscleActivationConfig, *, weld_toe_joints: bool=False) -> osim.ModelProcessor:
    import sys
    path_set = function_based_path_set_path() if bool(cfg.moco_use_function_based_paths) else None
    use_function_paths = bool(path_set is not None and path_set.is_file())
    if bool(cfg.moco_use_function_based_paths) and path_set is not None and not use_function_paths:
        print(f'WARNING: function-based path set missing at {path_set}; using geometry paths.', file=sys.stderr, flush=True)
    will_weld = bool(weld_toe_joints and cfg.moco_weld_toe_joints)
    if will_weld and use_function_paths and path_set is not None and function_based_path_set_depends_on_mtp(path_set):
        # Keep function-based paths; skip MTP welds so path coordinate dependencies remain valid.
        print(
            'WARNING: function-based path set references mtp_angle_*; skipping MTP welds so paths can connect. '
            'Re-fit paths on a welded-toe model to restore MocoTrack toe welding.',
            file=sys.stderr,
            flush=True,
        )
        will_weld = False
    mp = osim.ModelProcessor(str(base_model_path))
    if will_weld:
        joints_to_weld = osim.StdVectorString()
        for joint_name in _MOCO_TOE_JOINTS:
            joints_to_weld.append(joint_name)
        mp.append(osim.ModOpReplaceJointsWithWelds(joints_to_weld))
    mp.append(osim.ModOpIgnoreTendonCompliance())
    mp.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    mp.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    mp.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
    mp.append(osim.ModOpAddResiduals(float(cfg.moco_residual_force), float(cfg.moco_residual_force) * 0.2, 1.0))
    reserve_force = float(cfg.moco_reserve_optimal_force) * float(cfg.moco_reserve_scale)
    mp.append(osim.ModOpAddReserves(reserve_force, 1.0))
    if use_function_paths and path_set is not None:
        mp.append(osim.ModOpReplacePathsWithFunctionBasedPaths(str(path_set)))
    return mp

def muscle_names_from_processor(mp: osim.ModelProcessor) -> Tuple[str, ...]:
    model = mp.process()
    model.initSystem()
    muscles = model.getMuscles()
    return tuple((muscles.get(i).getName() for i in range(muscles.getSize())))

def prepare_rajagopal_activation_model(work_dir: Path, cfg: MuscleActivationConfig, *, weld_toe_joints: bool=False) -> Tuple[Path, Tuple[str, ...]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    tag = 'moco' if weld_toe_joints else 'static'
    out = work_dir / f'rajagopal_activation_{tag}.osim'
    base = prepare_unlocked_rajagopal_base(work_dir)
    mp = build_activation_model_processor(base, cfg, weld_toe_joints=weld_toe_joints)
    processed = mp.process()
    processed.initSystem()
    muscles = processed.getMuscles()
    names = tuple((muscles.get(i).getName() for i in range(muscles.getSize())))
    if not out.is_file():
        processed.printToXML(str(out))
    return (out, names)
