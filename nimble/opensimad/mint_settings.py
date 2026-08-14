"""MinT-aligned OpenSimAD / OpenCap tracking settings (Rajagopal @ 20 fps)."""
from __future__ import annotations
from typing import Any, Dict
from nimble.opensimad import OPENSIM_MODEL_BASENAME

# MinT appendix A.3: 50 collocation points/s, tol 1e-3, 2500 iter cap, 1.4s/0.14s windows.
MINT_MESH_DENSITY = 50
MINT_CONVERGENCE_TOLERANCE = 1e-3
MINT_MAX_ITERATIONS = 2500
MINT_CORE_DURATION_S = 1.4
MINT_BUFFER_DURATION_S = 0.14
MINT_PARALLEL_SEGMENTS = 6
# OpenCap uses ipopt_tolerance as decade exponent: tol = 10**(-ipopt_tolerance).
# 1e-3 => ipopt_tolerance=3.
MINT_IPOPT_TOLERANCE = 3

def mint_tracking_settings(*, mass_kg: float=70.0, height_m: float=1.75, trial_name: str='segment') -> Dict[str, Any]:
    """OpenCap get_setup('other') tuned to MinT mesh/tol/iters; MTP welded off."""
    return {
        'OpenSimModel': OPENSIM_MODEL_BASENAME,
        'trial_name': trial_name,
        'mass_kg': float(mass_kg),
        'height_m': float(height_m),
        'treadmill_speed': 0,
        'contact_side': 'all',
        'useExpressionGraphFunction': True,
        'withMTP': False,
        'withArms': True,
        'withLumbarCoordinateActuators': True,
        'ipopt_tolerance': int(MINT_IPOPT_TOLERANCE),
        'max_iterations': int(MINT_MAX_ITERATIONS),
        'meshDensity': int(MINT_MESH_DENSITY),
        'ignorePassiveFiberForce': True,
        'filter_Qs_toTrack': True,
        'cutoff_freq_Qs': 6,
        'filter_Qds_toTrack': True,
        'cutoff_freq_Qds': 6,
        'filter_Qdds_toTrack': True,
        'cutoff_freq_Qdds': 6,
        'splineQds': True,
        'yCalcnToes': True,
        'weights': {
            'positionTrackingTerm': 100,
            'velocityTrackingTerm': 10,
            'accelerationTrackingTerm': 50,
            'activationTerm': 10,
            'armExcitationTerm': 0.001,
            'lumbarExcitationTerm': 0.001,
            'jointAccelerationTerm': 0.001,
            'activationDtTerm': 0.001,
            'forceDtTerm': 0.001,
        },
        'coordinates_toTrack': {
            'pelvis_tilt': {'weight': 10},
            'pelvis_list': {'weight': 10},
            'pelvis_rotation': {'weight': 10},
            'pelvis_tx': {'weight': 10},
            'pelvis_ty': {'weight': 10},
            'pelvis_tz': {'weight': 10},
            'hip_flexion_l': {'weight': 20},
            'hip_adduction_l': {'weight': 10},
            'hip_rotation_l': {'weight': 1},
            'hip_flexion_r': {'weight': 20},
            'hip_adduction_r': {'weight': 10},
            'hip_rotation_r': {'weight': 1},
            'knee_angle_l': {'weight': 10},
            'knee_angle_r': {'weight': 10},
            'ankle_angle_l': {'weight': 10},
            'ankle_angle_r': {'weight': 10},
            'subtalar_angle_l': {'weight': 10},
            'subtalar_angle_r': {'weight': 10},
            'lumbar_extension': {'weight': 10},
            'lumbar_bending': {'weight': 10},
            'lumbar_rotation': {'weight': 10},
            'arm_flex_l': {'weight': 10},
            'arm_add_l': {'weight': 10},
            'arm_rot_l': {'weight': 10},
            'arm_flex_r': {'weight': 10},
            'arm_add_r': {'weight': 10},
            'arm_rot_r': {'weight': 10},
            'elbow_flex_l': {'weight': 10},
            'elbow_flex_r': {'weight': 10},
            'pro_sup_l': {'weight': 10},
            'pro_sup_r': {'weight': 10},
        },
        'coordinate_constraints': {'pelvis_tx': {'env_bound': 0.1}},
    }
