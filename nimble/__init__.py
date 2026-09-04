from __future__ import annotations

# Keep package import light (avoid circular imports / heavy OpenSim at import).
from nimble.physics import NIMBLE_AVAILABLE

__all__ = [
    'DeterministicNimbleGuidance',
    'NimbleGuidanceWeights',
    'NimbleGuidanceConfig',
    'OpenSimGuidance',
    'OpenSimGuidanceConfig',
    'build_nimble_guidance',
    'build_opensim_guidance',
    'MuscleActivationConfig',
    'compute_muscle_activation',
    'configure_opensim_logging',
    'NIMBLE_AVAILABLE',
    'fit_q',
]


def __getattr__(name: str):
    if name in {
        'DeterministicNimbleGuidance',
        'NimbleGuidanceWeights',
        'NimbleGuidanceConfig',
        'OpenSimGuidance',
        'OpenSimGuidanceConfig',
        'build_nimble_guidance',
        'build_opensim_guidance',
    }:
        from nimble import guidance as _g
        return getattr(_g, name)
    if name in {'MuscleActivationConfig', 'compute_muscle_activation', 'configure_opensim_logging'}:
        from nimble import muscle_activation as _m
        return getattr(_m, name)
    if name == 'fit_q':
        from nimble.opensim_ik import fit_q_lai
        return fit_q_lai
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
