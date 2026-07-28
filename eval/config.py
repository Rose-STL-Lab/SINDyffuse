"""HumanML3D evaluation configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.io import load_json
from common.paths import default_humanml3d_root, repo_root


def default_evaluator_root() -> Path:
    explicit = os.environ.get("SINDYFFUSE_EVALUATOR_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    bundled = repo_root() / "eval" / "checkpoints"
    if (bundled / "t2m" / "text_mot_match" / "model" / "finest.tar").is_file():
        return bundled
    fallback = Path("/mnt/Wenhao/.runtime/evaluator")
    if (fallback / "t2m" / "text_mot_match" / "model" / "finest.tar").is_file():
        return fallback
    return bundled


@dataclass
class EvalConfig:
    data_root: str = ""
    split: str = "test"
    batch_size: int = 32
    replication_times: int = 1
    diversity_times: int = 300
    max_text_len: int = 20
    unit_length: int = 4
    max_motion_length: int = 196
    min_motion_len: int = 40
    dim_pose: int = 263
    joints_num: int = 22
    device: str = "auto"
    evaluator_root: str = ""
    seed: int = 42

    def resolve(self) -> "EvalConfig":
        if not self.data_root:
            self.data_root = default_humanml3d_root()
        if not self.evaluator_root:
            self.evaluator_root = str(default_evaluator_root())
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> "EvalConfig":
        payload = load_json(str(path))
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in payload.items() if k in known}
        return cls(**kwargs).resolve()


@dataclass
class MethodSpec:
    name: str
    motion_dir: str
    format: str = "hml263"  # hml263 | mint_q

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MethodSpec":
        return cls(
            name=str(payload["name"]),
            motion_dir=str(payload["motion_dir"]),
            format=str(payload.get("format", "hml263")),
        )


@dataclass
class BaselineEvalConfig:
    methods: list[MethodSpec] = field(default_factory=list)
    include_ground_truth: bool = True
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "BaselineEvalConfig":
        payload = load_json(str(path))
        methods = [MethodSpec.from_dict(m) for m in payload.get("methods", [])]
        eval_fields = EvalConfig.__dataclass_fields__  # type: ignore[attr-defined]
        if "eval" in payload:
            eval_cfg = EvalConfig(**{**EvalConfig().__dict__, **payload["eval"]}).resolve()
        elif any(k in payload for k in eval_fields):
            eval_cfg = EvalConfig(**{k: payload[k] for k in eval_fields if k in payload}).resolve()
        else:
            eval_cfg = EvalConfig().resolve()
        return cls(
            methods=methods,
            include_ground_truth=bool(payload.get("include_ground_truth", True)),
            eval=eval_cfg,
        )


@dataclass
class GuidanceVariant:
    name: str
    guidance: str = "sindy"
    opt_steps: int = 0
    opt_lr: float = 1e-3
    cfg_scale: float = 2.5
    lambda_sindy: float | None = None
    checkpoint: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GuidanceVariant":
        return cls(
            name=str(payload["name"]),
            guidance=str(payload.get("guidance", "sindy")),
            opt_steps=int(payload.get("opt_steps", 0)),
            opt_lr=float(payload.get("opt_lr", 1e-3)),
            cfg_scale=float(payload.get("cfg_scale", 2.5)),
            lambda_sindy=payload.get("lambda_sindy"),
            checkpoint=str(payload.get("checkpoint", "")),
        )


@dataclass
class AblationEvalConfig:
    variants: list[GuidanceVariant] = field(default_factory=list)
    diffusion_checkpoint: str = ""
    sindy_checkpoint_dir: str = ""
    surrogate_checkpoint_dir: str = ""
    output_root: str = "results/eval/ablation"
    num_samples_per_caption: int = 1
    max_motions: int = 0
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "AblationEvalConfig":
        payload = load_json(str(path))
        variants = [GuidanceVariant.from_dict(v) for v in payload.get("variants", [])]
        eval_payload = payload.get("eval", {})
        eval_cfg = EvalConfig(**{**EvalConfig().__dict__, **eval_payload}).resolve()
        return cls(
            variants=variants,
            diffusion_checkpoint=str(payload.get("diffusion_checkpoint", "")),
            sindy_checkpoint_dir=str(payload.get("sindy_checkpoint_dir", "")),
            surrogate_checkpoint_dir=str(payload.get("surrogate_checkpoint_dir", "")),
            output_root=str(payload.get("output_root", "results/eval/ablation")),
            num_samples_per_caption=int(payload.get("num_samples_per_caption", 1)),
            max_motions=int(payload.get("max_motions", 0)),
            eval=eval_cfg,
        )
