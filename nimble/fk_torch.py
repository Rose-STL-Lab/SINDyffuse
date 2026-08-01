from __future__ import annotations
from typing import Tuple
import torch
from nimble.rajagopal_kin import body_origins, foot_body_indices, keypoints_torch
forward_kinematics = keypoints_torch
__all__ = ['forward_kinematics', 'body_origins', 'foot_body_indices', 'keypoints_torch']