"""Kinematic contact-force proxies from HumanML3D poses."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from osim.kinematics import finite_diff, safe_norm


def estimate_contact_from_hml3d(
    poses: np.ndarray,
    com_acc: np.ndarray,
    dt: float,
    mass_kg: float = 70.0,
    g_mps2: float = 9.81,
) -> Dict[str, np.ndarray]:
    # HumanML3D indexing convention in this project.
    left_foot = poses[:, 10, :]
    right_foot = poses[:, 11, :]
    left_vel = finite_diff(left_foot, dt)
    right_vel = finite_diff(right_foot, dt)
    ground_y = float(np.percentile(np.minimum(left_foot[:, 1], right_foot[:, 1]), 5))
    l_contact = ((left_foot[:, 1] - ground_y) < 0.06) & (np.linalg.norm(left_vel[:, [0, 2]], axis=1) < 1.2)
    r_contact = ((right_foot[:, 1] - ground_y) < 0.06) & (np.linalg.norm(right_vel[:, [0, 2]], axis=1) < 1.2)
    contact_lr = np.stack([l_contact, r_contact], axis=1).astype(np.float32)
    any_contact = (np.sum(contact_lr, axis=1, keepdims=True) > 0).astype(np.float32)

    # Kinematics-derived proxy; provenance tag lives in feature bank.
    g_vec = np.array([0.0, -g_mps2, 0.0], dtype=np.float32)
    f_total = mass_kg * (com_acc - g_vec.reshape(1, 3))
    f_total[:, 1] = np.maximum(f_total[:, 1], 0.0)
    f_total = f_total * any_contact
    f_l = f_total * contact_lr[:, 0:1]
    f_r = f_total * contact_lr[:, 1:2]
    denom = np.clip(contact_lr[:, 0:1] + contact_lr[:, 1:2], 1.0, None)
    f_l = f_l / denom
    f_r = f_r / denom
    tau_l = np.zeros_like(f_l)
    tau_r = np.zeros_like(f_r)
    wrench = np.concatenate([tau_l, f_l, tau_r, f_r], axis=1).astype(np.float32)
    return {
        "contact_wrench": wrench,
        "contact_wrench_l2": safe_norm(wrench),
        "contact_active": (safe_norm(wrench) > 1e-6).astype(np.float32),
        "summary_contact_ratio": np.mean((safe_norm(wrench) > 1e-6).astype(np.float32), axis=0, keepdims=True).astype(
            np.float32
        ),
        "inferred_contact_active_lr": contact_lr.astype(np.float32),
    }
