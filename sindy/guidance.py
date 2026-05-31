"""Learned SINDy guidance: sparse text-conditioned model of Nimble L_bio."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import clip  # type: ignore
import joblib
import numpy as np
import torch

from common.paths import nimble_b3d_dir
from nimble.channels import BIOMECH_COMPONENT_KEYS
from nimble.guidance import NimbleGuidanceConfig
from nimble.physics import load_model, physics_from_q, physics_from_q_batch
from sindy.library import ThetaLibrary, ThetaSpec
from sindy.model import TextToXi, predict_from_xi
from sindy.features import features_from_q_torch


def _install_numpy_pickle_compat() -> None:
    try:
        import numpy._core as np_core_mod  # type: ignore
    except Exception:
        np_core_mod = np.core  # type: ignore[attr-defined]
    sys.modules.setdefault("numpy._core", np_core_mod)


def _make_theta_spec(theta_tier: str, include_u: bool, include_c: bool, u_names: List[str]) -> ThetaSpec:
    return ThetaSpec(
        tier=theta_tier,
        include_bias=True,
        include_linear_u=bool(include_u),
        include_linear_c=bool(include_c),
        include_u_times_c=bool(include_u and include_c),
        include_phase_terms=bool(include_u and ("phase_sin" in u_names)),
        include_contact_gated=bool(include_c and str(theta_tier).strip().lower() == "tier3_contact_periodic"),
        max_cross_terms=30,
    )


def _load_train_config(sild_dir: Path) -> Dict[str, Any]:
    path = sild_dir / "sild_text_train_results.json"
    defaults = {"theta_tier": "tier2_moderate", "include_u": True, "include_c": True}
    if not path.is_file():
        return dict(defaults)
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        cfg = meta.get("config") or {}
        return {
            "theta_tier": str(cfg.get("theta_tier", defaults["theta_tier"])),
            "include_u": bool(cfg.get("include_u", defaults["include_u"])),
            "include_c": bool(cfg.get("include_c", defaults["include_c"])),
        }
    except Exception:
        return dict(defaults)


class LearnedSINDyGuidance:
    """Sparse SINDy model: ``pred = Θ(motion) · Ξ(text)`` trained to match per-frame L_bio.

    Diffusion guidance minimizes MSE between the SINDy prediction and **actual** L_bio from
    ``physics_from_q`` (FK-only, no train-time IK).
    """

    def __init__(
        self,
        sild_dir: str,
        data_root: str,
        fps: float = 20.0,
        clip_model_name: str = "ViT-B/32",
        *,
        bio_physics: NimbleGuidanceConfig | None = None,
    ):
        _install_numpy_pickle_compat()
        self.sild_dir = Path(sild_dir)
        self.data_root = Path(data_root)
        cache = nimble_b3d_dir(self.data_root)
        if not cache.is_dir():
            raise FileNotFoundError(f"SINDy guidance requires Nimble B3D cache at {cache}.")
        self.fps = float(fps)
        self._sk = load_model().skeleton
        self.bio_physics = bio_physics or NimbleGuidanceConfig(
            max_physics_frames=64,
            physics_on_cpu=False,
            fk_backend="torch",
        )

        ckpt = torch.load(self.sild_dir / "text_to_xi.pt", map_location="cpu")
        self.num_experts = int(ckpt.get("num_experts", 1))
        self.max_seq_len = int(ckpt.get("max_seq_len", 256))
        self.in_dim = int(ckpt["in_dim"])
        self.theta_dim = int(ckpt["theta_dim"])
        self.target_dim = int(ckpt["target_dim"])

        bio_names = list(ckpt.get("bio_channel_names") or BIOMECH_COMPONENT_KEYS)
        if len(bio_names) != len(BIOMECH_COMPONENT_KEYS):
            raise ValueError(
                f"Checkpoint bio_channel_names length {len(bio_names)} != "
                f"{len(BIOMECH_COMPONENT_KEYS)}; retrain with current SINDyffuse."
            )
        self.bio_channel_names = bio_names

        if self.target_dim != len(BIOMECH_COMPONENT_KEYS):
            raise ValueError(
                f"Checkpoint target_dim={self.target_dim} != {len(BIOMECH_COMPONENT_KEYS)}; "
                "retrain SINDy on L_bio targets."
            )

        self.text_to_xi = TextToXi(
            in_dim=self.in_dim,
            hidden_dim=int(ckpt["hidden_dim"]),
            theta_dim=self.theta_dim,
            target_dim=self.target_dim,
            num_experts=self.num_experts,
            max_seq_len=self.max_seq_len,
        )
        self.text_to_xi.load_state_dict(ckpt["model_state"])
        self.text_to_xi.eval()

        self.theta_scaler = joblib.load(self.sild_dir / "scaler_theta.pkl")
        self.y_scaler = joblib.load(self.sild_dir / "scaler_y.pkl")
        self.emb_scaler = None
        emb_path = self.sild_dir / "scaler_text_embed.pkl"
        if emb_path.is_file():
            self.emb_scaler = joblib.load(emb_path)

        self.mean = torch.tensor(np.load(cache / "Mean.npy").astype(np.float32))
        self.std = torch.tensor(np.load(cache / "Std.npy").astype(np.float32))
        self.clip_model_name = str(clip_model_name)
        self._clip_model = None
        self._clip_device: Optional[str] = None

        ndof = int(self._sk.getNumDofs())
        _, _, u_names_ref, c_names_ref = features_from_q_torch(
            torch.zeros(1, 8, ndof), self._sk, fps=self.fps
        )
        tcfg = _load_train_config(self.sild_dir)
        self._theta_u_names = list(u_names_ref)
        self._theta_c_names = list(c_names_ref)
        self._theta_include_u = bool(tcfg["include_u"])
        self._theta_include_c = bool(tcfg["include_c"])
        self._theta_library = ThetaLibrary(
            spec=_make_theta_spec(
                str(tcfg["theta_tier"]),
                self._theta_include_u,
                self._theta_include_c,
                self._theta_u_names,
            )
        )

    def _get_clip(self, device: torch.device):
        if self._clip_model is None or self._clip_device != str(device):
            model, _ = clip.load(self.clip_model_name, device=device, jit=False)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            self._clip_model = model
            self._clip_device = str(device)
        return self._clip_model

    def _embed_text(self, captions: List[str], device: torch.device) -> torch.Tensor:
        model = self._get_clip(device)
        with torch.no_grad():
            toks = clip.tokenize(captions, truncate=True).to(device)
            emb = model.encode_text(toks).float().detach().cpu().numpy().astype(np.float32)
        if emb.shape[1] != self.in_dim:
            if emb.shape[1] > self.in_dim:
                emb = emb[:, : self.in_dim]
            else:
                emb = np.pad(emb, ((0, 0), (0, self.in_dim - emb.shape[1])))
        if self.emb_scaler is not None and getattr(self.emb_scaler, "n_features_in_", emb.shape[1]) == emb.shape[1]:
            emb = self.emb_scaler.transform(emb).astype(np.float32)
        return torch.tensor(emb, dtype=torch.float32, device=device)

    def _denorm_motion(self, motion_norm: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(motion_norm.device).view(1, 1, -1)
        std = self.std.to(motion_norm.device).view(1, 1, -1)
        return motion_norm * std + mean

    def _scale_theta(self, theta: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.theta_scaler.mean_.astype(np.float32), device=theta.device).view(1, 1, -1)
        scale = torch.tensor(self.theta_scaler.scale_.astype(np.float32), device=theta.device).view(1, 1, -1)
        return (theta - mean) / torch.clamp(scale, min=1e-8)

    def _scale_y(self, y: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.y_scaler.mean_.astype(np.float32), device=y.device).view(1, 1, -1)
        scale = torch.tensor(self.y_scaler.scale_.astype(np.float32), device=y.device).view(1, 1, -1)
        return (y - mean) / torch.clamp(scale, min=1e-8)

    def _build_theta(self, motion_norm: torch.Tensor) -> torch.Tensor:
        b, t, _ = motion_norm.shape
        denorm = self._denorm_motion(motion_norm)
        use_fk = str(getattr(self.bio_physics, "fk_backend", "torch")).strip().lower() == "torch"
        u_t, c_t, _, _ = features_from_q_torch(
            denorm, self._sk, fps=self.fps, use_torch_fk=use_fk
        )
        u_in = u_t[:, :-1, :] if self._theta_include_u else None
        c_in = c_t[:, :-1, :] if self._theta_include_c else None
        theta_flat, _ = self._theta_library.build_torch(
            u=u_in,
            c=c_in,
            u_names=self._theta_u_names if self._theta_include_u else [],
            c_names=self._theta_c_names if self._theta_include_c else [],
        )
        theta = theta_flat.view(b, t - 1, -1)
        if theta.shape[-1] != self.theta_dim:
            raise ValueError(f"Theta dim mismatch: built {theta.shape[-1]} vs ckpt {self.theta_dim}")
        return theta

    def _bio_from_motion(self, motion_norm: torch.Tensor) -> torch.Tensor:
        """Per-sequence L_bio ``[B, L, C]`` from predicted motion (for guidance MSE)."""
        b, t, _ = motion_norm.shape
        denorm = self._denorm_motion(motion_norm)
        dt = 1.0 / max(self.fps, 1e-8)
        comp_list = physics_from_q_batch(
            denorm,
            guidance_cfg=self.bio_physics,
            dt=float(dt),
            fps=self.fps,
        )
        rows: List[torch.Tensor] = []
        for comp in comp_list:
            cols = [comp[k].reshape(-1) for k in self.bio_channel_names]
            bio = torch.stack(cols, dim=-1)
            if bio.shape[0] > t - 1:
                bio = bio[: t - 1]
            rows.append(bio.detach())
        return torch.stack(rows, dim=0)

    def loss(self, motion_norm: torch.Tensor, captions: List[str], device: torch.device) -> torch.Tensor:
        b, t, _ = motion_norm.shape
        if t < 3:
            return torch.tensor(0.0, device=device)

        theta_s = self._scale_theta(self._build_theta(motion_norm))
        emb = self._embed_text(captions, device=device)
        self.text_to_xi = self.text_to_xi.to(device)
        xi_out = self.text_to_xi(emb, seq_len=int(t - 1))
        pred_s = predict_from_xi(theta_s, xi_out)
        y_s = self._scale_y(self._bio_from_motion(motion_norm))
        return torch.mean((pred_s - y_s) ** 2)

    def loss_and_stats(self, motion_norm: torch.Tensor, captions: List[str], device: torch.device) -> tuple[torch.Tensor, Dict[str, float]]:
        loss = self.loss(motion_norm, captions, device)
        return loss, {"sindy_guidance_scalar": float(loss.detach().cpu().item())}
