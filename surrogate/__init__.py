"""Differentiable muscle-activation surrogate and OpenSim ground-truth pipeline."""

from __future__ import annotations

from surrogate.opensim_activation import (
    MuscleActivationConfig,
    MuscleActivationResult,
    activation_stats,
    compute_muscle_activation,
    muscle_names,
    rajagopal_model_path,
)
from surrogate.model import ActivationSurrogate, build_activation_surrogate
from surrogate.guidance import ActivationSurrogateGuidance, load_activation_surrogate_guidance

__all__ = [
    "MuscleActivationConfig",
    "MuscleActivationResult",
    "compute_muscle_activation",
    "activation_stats",
    "muscle_names",
    "rajagopal_model_path",
    "ActivationSurrogate",
    "build_activation_surrogate",
    "ActivationSurrogateGuidance",
    "load_activation_surrogate_guidance",
]
