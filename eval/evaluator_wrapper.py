# Adapted from T2M (https://github.com/EricGuo5513/text-to-motion).
# Copyright (c) 2022 Chuan Guo

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from eval.evaluator_modules import MotionEncoderBiGRUCo, MovementConvEncoder, TextEncoderBiGRUCo


@dataclass
class EvaluatorConfig:
    dataset_name: str = "t2m"
    dim_pose: int = 263
    dim_word: int = 300
    dim_pos_ohot: int = 15
    dim_text_hidden: int = 512
    dim_motion_hidden: int = 1024
    dim_coemb_hidden: int = 512
    dim_movement_enc_hidden: int = 512
    dim_movement_latent: int = 512
    unit_length: int = 4
    max_motion_length: int = 196
    evaluator_dir: str = ""
    glove_dir: str = ""
    device: str = "cpu"

    @classmethod
    def from_paths(cls, evaluator_root: str | Path, device: str = "cpu") -> "EvaluatorConfig":
        root = Path(evaluator_root).resolve()
        return cls(
            evaluator_dir=str(root),
            glove_dir=str(root / "glove"),
            device=device,
        )


def build_evaluator_models(cfg: EvaluatorConfig) -> tuple[TextEncoderBiGRUCo, MotionEncoderBiGRUCo, MovementConvEncoder]:
    movement_enc = MovementConvEncoder(
        cfg.dim_pose - 4,
        cfg.dim_movement_enc_hidden,
        cfg.dim_movement_latent,
    )
    text_enc = TextEncoderBiGRUCo(
        word_size=cfg.dim_word,
        pos_size=cfg.dim_pos_ohot,
        hidden_size=cfg.dim_text_hidden,
        output_size=cfg.dim_coemb_hidden,
        device=cfg.device,
    )
    motion_enc = MotionEncoderBiGRUCo(
        input_size=cfg.dim_movement_latent,
        hidden_size=cfg.dim_motion_hidden,
        output_size=cfg.dim_coemb_hidden,
        device=cfg.device,
    )
    ckpt_path = (
        Path(cfg.evaluator_dir)
        / cfg.dataset_name
        / "text_mot_match"
        / "model"
        / "finest.tar"
    )
    checkpoint = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
    movement_enc.load_state_dict(checkpoint["movement_encoder"])
    text_enc.load_state_dict(checkpoint["text_encoder"])
    motion_enc.load_state_dict(checkpoint["motion_encoder"])
    print(f"Loaded HumanML3D evaluator (epoch {checkpoint['epoch']}) from {ckpt_path}")
    return text_enc, motion_enc, movement_enc


class EvaluatorModelWrapper:
    def __init__(self, cfg: EvaluatorConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.text_encoder, self.motion_encoder, self.movement_encoder = build_evaluator_models(cfg)
        self.text_encoder.to(self.device)
        self.motion_encoder.to(self.device)
        self.movement_encoder.to(self.device)
        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    @torch.no_grad()
    def get_co_embeddings(
        self,
        word_embs: torch.Tensor,
        pos_ohot: torch.Tensor,
        cap_lens: torch.Tensor,
        motions: torch.Tensor,
        m_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        word_embs = word_embs.detach().to(self.device).float()
        pos_ohot = pos_ohot.detach().to(self.device).float()
        motions = motions.detach().to(self.device).float()

        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]

        movements = self.movement_encoder(motions[..., :-4]).detach()
        m_lens_enc = torch.div(m_lens, self.cfg.unit_length, rounding_mode="trunc")
        motion_embedding = self.motion_encoder(movements, m_lens_enc)
        text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
        text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    @torch.no_grad()
    def get_motion_embeddings(self, motions: torch.Tensor, m_lens: torch.Tensor) -> torch.Tensor:
        motions = motions.detach().to(self.device).float()
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]
        movements = self.movement_encoder(motions[..., :-4]).detach()
        m_lens_enc = torch.div(m_lens, self.cfg.unit_length, rounding_mode="trunc")
        return self.motion_encoder(movements, m_lens_enc)
