from __future__ import annotations
import shutil
from pathlib import Path
from typing import Tuple
import opensim as osim
import numpy as np
from nimble.muscle_activation import MuscleActivationConfig, opensim_quiet, rajagopal_model_path
from nimble.opensimad.paths import ad_base_model_path, ad_contacts_model_path, ad_scaled_adjusted_model_path, opensimad_dir
from nimble.rajagopal_model import _MOCO_TOE_JOINTS, function_based_path_set_path, unlock_rajagopal_coordinates

# OpenCap-style multi-sphere foot contacts (utilsProcessing.generate_model_with_contacts).
_REFERENCE_CONTACT_SPHERES = {
    's1_r': {'radius': 0.032, 'location': np.array([0.0019011578840796601, -0.01, -0.00382630379623308]), 'socket_frame': 'calcn_r'},
    's2_r': {'radius': 0.032, 'location': np.array([0.14838639994206301, -0.01, -0.028713422052654002]), 'socket_frame': 'calcn_r'},
    's3_r': {'radius': 0.032, 'location': np.array([0.13300117060705099, -0.01, 0.051636247344956601]), 'socket_frame': 'calcn_r'},
    's4_r': {'radius': 0.032, 'location': np.array([0.066234666199163503, -0.01, 0.026364160674169801]), 'socket_frame': 'calcn_r'},
    's5_r': {'radius': 0.032, 'location': np.array([0.06, -0.01, -0.018760308461917698]), 'socket_frame': 'toes_r'},
    's6_r': {'radius': 0.032, 'location': np.array([0.045, -0.01, 0.061856956754965199]), 'socket_frame': 'toes_r'},
    's1_l': {'radius': 0.032, 'location': np.array([0.0019011578840796601, -0.01, 0.00382630379623308]), 'socket_frame': 'calcn_l'},
    's2_l': {'radius': 0.032, 'location': np.array([0.14838639994206301, -0.01, 0.028713422052654002]), 'socket_frame': 'calcn_l'},
    's3_l': {'radius': 0.032, 'location': np.array([0.13300117060705099, -0.01, -0.051636247344956601]), 'socket_frame': 'calcn_l'},
    's4_l': {'radius': 0.032, 'location': np.array([0.066234666199163503, -0.01, -0.026364160674169801]), 'socket_frame': 'calcn_l'},
    's5_l': {'radius': 0.032, 'location': np.array([0.06, -0.01, 0.018760308461917698]), 'socket_frame': 'toes_l'},
    's6_l': {'radius': 0.032, 'location': np.array([0.045, -0.01, -0.061856956754965199]), 'socket_frame': 'toes_l'},
}

def prepare_welded_unlocked_rajagopal(work_dir: Path | None=None, *, force: bool=False) -> Path:
    """Unlocked coordinates + MTP joints welded (OpenSimAD-compatible; no locked DOFs)."""
    out_dir = Path(work_dir) if work_dir is not None else opensimad_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'rajagopal_unlocked_mtp_welded.osim'
    if out.is_file() and not force:
        return out
    with opensim_quiet('Off'):
        model = osim.Model(str(rajagopal_model_path()))
        unlock_rajagopal_coordinates(model)
        joints = osim.StdVectorString()
        for name in _MOCO_TOE_JOINTS:
            joints.append(name)
        # ReplaceJointsWithWelds via ModelProcessor for robustness across OpenSim versions.
        tmp = out_dir / '_tmp_unlocked.osim'
        model.initSystem()
        model.printToXML(str(tmp))
        mp = osim.ModelProcessor(str(tmp))
        mp.append(osim.ModOpReplaceJointsWithWelds(joints))
        welded = mp.process()
        welded.initSystem()
        welded.printToXML(str(out))
        tmp.unlink(missing_ok=True)
    return out

def _replace_simm_splines_in_spatial_transforms(model: osim.Model) -> None:
    """Apply the MinT/OpenSimAD spline compatibility preprocessing.

    OpenSimAD's generated model supports PolynomialFunction but not SimmSpline.
    Fit a degree-five polynomial to each spatial-transform spline, matching the
    upstream MinT/OpenSimAD model-preparation convention.
    """
    converted = []
    for index in range(model.get_JointSet().getSize()):
        joint = model.get_JointSet().get(index)
        if joint.getConcreteClassName() != 'CustomJoint':
            continue
        custom_joint = osim.CustomJoint.safeDownCast(joint)
        spatial_transform = custom_joint.get_SpatialTransform()
        axes = (
            ('rotation1', spatial_transform.get_rotation1()),
            ('rotation2', spatial_transform.get_rotation2()),
            ('rotation3', spatial_transform.get_rotation3()),
            ('translation1', spatial_transform.get_translation1()),
            ('translation2', spatial_transform.get_translation2()),
            ('translation3', spatial_transform.get_translation3()),
        )
        for axis_name, axis in axes:
            function = axis.get_function()
            if function.getConcreteClassName() != 'SimmSpline':
                continue
            spline = osim.SimmSpline.safeDownCast(function)
            x = spline.getX().to_numpy()
            y = spline.getY().to_numpy()
            if x.shape[0] < 2:
                raise ValueError(f'{joint.getName()} {axis_name} SimmSpline has fewer than two points')
            degree = min(5, x.shape[0] - 1)
            coefficients = np.polynomial.polynomial.polyfit(x, y, degree)
            axis.set_function(osim.PolynomialFunction(osim.Vector(coefficients.tolist())))
            converted.append(f'{joint.getName()}.{axis_name}')
    if converted:
        print('OpenSimAD spline compatibility: replaced ' + ', '.join(converted))

def prepare_ad_base_model(*, force: bool=False) -> Path:
    """Write AD base model: welded MTP + function-based paths applied."""
    out = ad_base_model_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and not force:
        return out
    welded = prepare_welded_unlocked_rajagopal(out.parent, force=force)
    path_set = function_based_path_set_path()
    with opensim_quiet('Off'):
        mp = osim.ModelProcessor(str(welded))
        if path_set.is_file():
            mp.append(osim.ModOpReplacePathsWithFunctionBasedPaths(str(path_set)))
        model = mp.process()
        _replace_simm_splines_in_spatial_transforms(model)
        model.initSystem()
        model.printToXML(str(out))
    # OpenCap naming alias (no contacts yet).
    scaled = ad_scaled_adjusted_model_path()
    shutil.copy2(out, scaled)
    return out

def _add_opencap_contacts(model: osim.Model) -> None:
    ground = model.getGround()
    half = osim.ContactHalfSpace(osim.Vec3(0, 0, 0), osim.Vec3(0, 0, -np.pi / 2.0), ground, 'floor')
    model.addContactGeometry(half)
    stiffness = 1000000.0
    dissipation = 2.0
    for name, spec in _REFERENCE_CONTACT_SPHERES.items():
        body = model.getBodySet().get(spec['socket_frame'])
        loc = spec['location']
        sphere = osim.ContactSphere(float(spec['radius']), osim.Vec3(float(loc[0]), float(loc[1]), float(loc[2])), body, name)
        model.addContactGeometry(sphere)
        force = osim.SmoothSphereHalfSpaceForce(f'contact_{name}', sphere, half)
        force.set_stiffness(stiffness)
        force.set_dissipation(dissipation)
        force.set_static_friction(0.8)
        force.set_dynamic_friction(0.8)
        force.set_viscous_friction(0.5)
        force.set_transition_velocity(0.2)
        model.addForce(force)
    model.finalizeConnections()

def prepare_ad_contacts_model(*, force: bool=False) -> Path:
    """AD base + OpenCap multi-sphere contacts → OpenSimAD external-function input."""
    out = ad_contacts_model_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and not force:
        return out
    base = prepare_ad_base_model(force=force)
    with opensim_quiet('Off'):
        model = osim.Model(str(base))
        _add_opencap_contacts(model)
        model.initSystem()
        model.printToXML(str(out))
    return out

def ensure_ad_ready_artifacts(*, force: bool=False) -> Tuple[Path, Path]:
    base = prepare_ad_base_model(force=force)
    contacts = prepare_ad_contacts_model(force=force)
    return (base, contacts)
