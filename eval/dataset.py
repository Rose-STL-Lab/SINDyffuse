"""HumanML3D dataset for text-to-motion evaluation."""

from __future__ import annotations

import codecs
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.splits import load_split_ids
from eval.config import EvalConfig
from eval.word_vectorizer import WordVectorizer


@dataclass
class EvalSample:
    sample_id: str
    motion_id: str
    caption: str
    tokens: list[str]
    motion: np.ndarray
    length: int


class HumanML3DEvalDataset(Dataset):
    """Ground-truth or generated HumanML3D motions for evaluation."""

    def __init__(
        self,
        cfg: EvalConfig,
        *,
        motion_dir: Path | None = None,
        motion_lookup: dict[str, np.ndarray] | None = None,
        split: str | None = None,
    ):
        self.cfg = cfg.resolve()
        self.root = Path(self.cfg.data_root).resolve()
        self.motion_dir = motion_dir or (self.root / "new_joint_vecs")
        self.text_dir = self.root / "texts" if (self.root / "texts").is_dir() else self.root
        self.motion_lookup = motion_lookup
        self.w_vectorizer = WordVectorizer(Path(self.cfg.evaluator_root) / "glove")
        self.mean = np.load(Path(self.cfg.evaluator_root) / "t2m_mean.npy").astype(np.float32)
        self.std = np.load(Path(self.cfg.evaluator_root) / "t2m_std.npy").astype(np.float32)
        self.samples = self._build_samples(split or self.cfg.split)

    def _build_samples(self, split: str) -> list[EvalSample]:
        samples: list[EvalSample] = []
        for motion_id in load_split_ids(self.root, split):
            motion = self._load_motion(motion_id)
            if motion is None:
                continue
            text_path = self.text_dir / f"{motion_id}.txt"
            if not text_path.is_file():
                continue
            for sample in self._expand_captions(motion_id, motion, text_path):
                if sample.length >= self.cfg.min_motion_len and sample.length < 200:
                    samples.append(sample)
        if not samples:
            raise ValueError(f"No valid evaluation samples for split={split!r}")
        samples.sort(key=lambda s: s.length)
        return samples

    def _load_motion(self, motion_id: str) -> np.ndarray | None:
        if self.motion_lookup is not None:
            return self.motion_lookup.get(motion_id)
        path = self.motion_dir / f"{motion_id}.npy"
        if not path.is_file():
            return None
        return np.load(path).astype(np.float32)

    def _expand_captions(
        self,
        motion_id: str,
        motion: np.ndarray,
        text_path: Path,
    ) -> Iterator[EvalSample]:
        with codecs.open(text_path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        full_texts: list[tuple[str, list[str]]] = []
        for line in lines:
            parts = line.strip().split("#")
            if len(parts) < 4:
                continue
            caption = parts[0].strip()
            tokens = parts[1].split()
            f_tag = 0.0 if np.isnan(float(parts[2])) else float(parts[2])
            to_tag = 0.0 if np.isnan(float(parts[3])) else float(parts[3])
            if f_tag == 0.0 and to_tag == 0.0:
                full_texts.append((caption, tokens))
                continue
            seg = motion[int(f_tag * 20) : int(to_tag * 20)]
            if len(seg) < self.cfg.min_motion_len:
                continue
            sample_id = f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}_{motion_id}"
            yield EvalSample(
                sample_id=sample_id,
                motion_id=motion_id,
                caption=caption,
                tokens=tokens,
                motion=seg,
                length=len(seg),
            )
        if full_texts:
            for caption, tokens in full_texts:
                yield EvalSample(
                    sample_id=motion_id,
                    motion_id=motion_id,
                    caption=caption,
                    tokens=tokens,
                    motion=motion,
                    length=len(motion),
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _prepare_motion(self, motion: np.ndarray, m_length: int) -> tuple[np.ndarray, int]:
        if self.cfg.unit_length < 10:
            coin = random.choice(["single", "single", "double"])
        else:
            coin = "single"
        if coin == "double":
            m_length = (m_length // self.cfg.unit_length - 1) * self.cfg.unit_length
        else:
            m_length = (m_length // self.cfg.unit_length) * self.cfg.unit_length
        m_length = max(m_length, self.cfg.min_motion_len)
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx : idx + m_length]
        motion = (motion - self.mean) / self.std
        if m_length < self.cfg.max_motion_length:
            motion = np.concatenate(
                [motion, np.zeros((self.cfg.max_motion_length - m_length, motion.shape[1]), dtype=np.float32)],
                axis=0,
            )
        return motion.astype(np.float32), m_length

    def _tokenize(self, tokens: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
        if len(tokens) < self.cfg.max_text_len:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.cfg.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[: self.cfg.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        return (
            np.concatenate(word_embeddings, axis=0).astype(np.float32),
            np.concatenate(pos_one_hots, axis=0).astype(np.float32),
            sent_len,
        )

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        motion, m_length = self._prepare_motion(sample.motion.copy(), sample.length)
        word_embeddings, pos_one_hots, sent_len = self._tokenize(sample.tokens)
        return (
            torch.from_numpy(word_embeddings),
            torch.from_numpy(pos_one_hots),
            sample.caption,
            sent_len,
            torch.from_numpy(motion),
            m_length,
            sample.sample_id,
        )


def collate_eval_batch(batch):
    word_embeddings, pos_one_hots, captions, sent_lens, motions, m_lengths, sample_ids = zip(*batch)
    return (
        torch.stack(word_embeddings),
        torch.stack(pos_one_hots),
        list(captions),
        torch.tensor(sent_lens, dtype=torch.long),
        torch.stack(motions),
        torch.tensor(m_lengths, dtype=torch.long),
        list(sample_ids),
    )


def build_eval_dataloader(
    dataset: HumanML3DEvalDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_eval_batch,
    )
