"""Lhotse-based DataLoader factory for CTC training and validation.

Inspired by IceFall's ``asr_datamodule.py`` but stripped down to the bits we
actually need for CTC. Key features:

  * Loads CutSets from ``cuts_{part}.jsonl.gz`` (produced by
    :func:`coe_ctc.data.fbank.compute_fbank`).
  * Dynamic bucketing (DynamicBucketingSampler) for efficient batching of
    variable-length utterances.
  * Per-utterance MVN on the GPU (fast, no precomputation).
  * SpecAugment applied at collate time on training batches only.
  * SentencePiece tokenization (lazy-loaded; same processor shared across
    epochs).

We do NOT use Lhotse's K2SpeechRecognitionDataset directly — that ties us
to k2; instead we ship a tiny ``CtcDataset`` and a manual collator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class DataModuleConfig:
    """All knobs the data side cares about. Mirrors IceFall conventions."""

    manifest_dir: str = ""
    feats_dir: str = ""            # informational only — features are referenced inside the cut set
    bpe_model: str = ""            # path to spm.model
    train_parts: Sequence[str] = ("train-clean-100",)
    dev_part: str = "dev-other"
    test_parts: Sequence[str] = ("test-clean", "test-other", "dev-clean", "dev-other")
    manifest_prefix: str = "librispeech"  # filename prefix used by cuts_*.jsonl.gz

    # Bucketing / batching
    max_duration: float = 200.0    # seconds per micro-batch (typical icefall value)
    num_buckets: int = 30
    shuffle: bool = True
    drop_last: bool = True

    # Filtering
    min_seconds: float = 1.0
    max_seconds: float = 20.0

    # Tokenizer
    blank_id: int = 0

    # SpecAugment
    apply_spec_augment: bool = True
    spec_aug_num_freq_masks: int = 2
    spec_aug_freq_mask_param: int = 27
    spec_aug_num_time_masks: int = 10
    spec_aug_time_mask_param: int = 100
    spec_aug_time_mask_ratio: float = 0.04

    # Normalization
    per_utt_mvn: bool = True

    # Dataloader
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


# ─────────────────────────────────────────────────────────────────────────
# Tokenizer wrapper
# ─────────────────────────────────────────────────────────────────────────


def _bpe_missing_hint(model_path: str) -> str:
    """Build a friendly error message that tells the user how to train the missing BPE."""
    import re

    p = Path(model_path)
    msg = [f"BPE model not found: {model_path}", ""]
    m = re.search(r"(libri\w+)_bpe(\d+)", str(p))
    if m:
        data_alias, vocab = m.group(1), m.group(2)
        # libri100 is a derived alias of libri960 — point users at libri960
        # so the symlink mirroring kicks in and both aliases work afterwards.
        prep_alias = "libri960" if data_alias == "libri100" else data_alias
        msg += [
            "Train it without re-downloading or re-extracting features:",
            f"    bash scripts/preprocess/run_preprocess.sh --data {prep_alias} "
            f"--n_bpe {vocab} \\",
            "        --skip-download --skip-manifest --skip-fbank",
        ]
    else:
        msg += [
            "Train a SentencePiece model with:",
            "    bash scripts/preprocess/run_preprocess.sh --data libri960 --n_bpe <V>",
            "or set data.bpe_model in your YAML to an existing model file.",
        ]
    return "\n".join(msg)


class BpeTokenizer:
    """Thin wrapper around SentencePiece for encode/decode."""

    def __init__(self, model_path: str) -> None:
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        p = Path(model_path)
        if not p.is_file():
            raise FileNotFoundError(_bpe_missing_hint(model_path))
        self.sp.Load(model_path)

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    def encode(self, text: str) -> list[int]:
        return self.sp.EncodeAsIds(text)

    def decode(self, ids: Sequence[int]) -> str:
        return self.sp.DecodeIds(list(ids))

    def id_to_piece(self, idx: int) -> str:
        return self.sp.IdToPiece(int(idx))

    def vocab_list(self) -> list[str]:
        return [self.sp.IdToPiece(i) for i in range(self.vocab_size)]


# ─────────────────────────────────────────────────────────────────────────
# Dataset + collator
# ─────────────────────────────────────────────────────────────────────────


class CtcDataset(Dataset):
    """Maps a Lhotse ``CutSet`` to dict batches consumable by the model.

    Each item materializes features for one cut and tokenizes its
    transcript. Batching/bucketing happens upstream via Lhotse's sampler.
    """

    def __init__(self, cuts, tokenizer: BpeTokenizer) -> None:
        self.cuts = cuts
        self.tokenizer = tokenizer
        # Single pass over (possibly lazy) cuts builds both the order and lookup table.
        self._ids: list[str] = []
        self._cut_lookup: dict = {}
        for c in cuts:
            self._ids.append(c.id)
            self._cut_lookup[c.id] = c

    @property
    def ids(self) -> list[str]:
        return self._ids

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> dict:
        cut = self._cut_lookup[self._ids[idx]]
        feats = cut.load_features()  # numpy (T, F)
        # Lhotse may return float32; convert to tensor.
        feats_t = torch.from_numpy(feats)
        text = " ".join(s.text or "" for s in cut.supervisions).strip()
        tokens = self.tokenizer.encode(text)
        return {
            "cut_id": cut.id,
            "features": feats_t,
            "feature_length": feats_t.size(0),
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "target_length": len(tokens),
            "text": text,
        }


def _collate_batch(batch: list[dict]) -> dict:
    """Pad features and tokens to the longest in the batch."""
    b = len(batch)
    max_t = max(item["feature_length"] for item in batch)
    max_u = max(max(item["target_length"], 1) for item in batch)
    f_dim = batch[0]["features"].size(1)

    features = torch.zeros(b, max_t, f_dim, dtype=torch.float32)
    feature_lengths = torch.zeros(b, dtype=torch.long)
    tokens = torch.zeros(b, max_u, dtype=torch.long)
    target_lengths = torch.zeros(b, dtype=torch.long)
    cut_ids: list[str] = []
    texts: list[str] = []

    for i, item in enumerate(batch):
        t = item["feature_length"]
        u = item["target_length"]
        features[i, :t] = item["features"]
        feature_lengths[i] = t
        if u > 0:
            tokens[i, :u] = item["tokens"]
        target_lengths[i] = max(u, 1)
        cut_ids.append(item["cut_id"])
        texts.append(item["text"])

    return {
        "cut_ids": cut_ids,
        "texts": texts,
        "features": features,
        "feature_lengths": feature_lengths,
        "tokens": tokens,
        "target_lengths": target_lengths,
    }


# ─────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────


class CtcDataModule:
    """Builds train/valid/test loaders from a config + manifest dir.

    Only train/valid loaders are used during training. ``test_loaders()``
    returns a dict for the evaluation entry point.
    """

    def __init__(self, cfg: DataModuleConfig) -> None:
        self.cfg = cfg
        self.tokenizer: Optional[BpeTokenizer] = None
        if cfg.bpe_model:
            self.tokenizer = BpeTokenizer(cfg.bpe_model)
        # Keep the unfiltered eager CutSet per part so curriculum rebuilds
        # only re-apply the duration filter (no JSONL re-parse).
        self._eager_cuts_cache: dict = {}
        self._last_clamp_warn: tuple[int, int] | None = None

    # ----------------------------------------------------------------- core
    def _eager_cuts(self, part: str):
        if part not in self._eager_cuts_cache:
            from lhotse import CutSet

            path = Path(self.cfg.manifest_dir) / f"cuts_{part}.jsonl.gz"
            if not path.exists():
                raise FileNotFoundError(
                    f"Expected cut manifest {path}. Did you run scripts/preprocess/run_preprocess.sh?"
                )
            self._eager_cuts_cache[part] = CutSet.from_jsonl_lazy(str(path)).to_eager()
        return self._eager_cuts_cache[part]

    def _load_cuts(self, part: str, *, max_seconds_override: Optional[float] = None):
        from coe_ctc.data.transforms import LengthFilter

        min_s = max(0.0, self.cfg.min_seconds)
        if max_seconds_override is not None:
            max_s = max(min_s + 1e-6, float(max_seconds_override))
        elif self.cfg.max_seconds > 0:
            max_s = self.cfg.max_seconds
        else:
            max_s = float("inf")
        return LengthFilter(min_seconds=min_s, max_seconds=max_s)(self._eager_cuts(part)).to_eager()

    def _make_sampler(self, cuts, *, shuffle: bool, drop_last: bool):
        from lhotse.dataset.sampling import DynamicBucketingSampler

        # Lhotse requires num_buckets ≤ len(cuts); curriculum's narrow first phase
        # (e.g. ≤2s on LibriSpeech) can leave fewer cuts than the configured buckets.
        n_cuts = len(cuts)
        n_buckets = max(1, min(self.cfg.num_buckets, n_cuts))
        if n_buckets < self.cfg.num_buckets and self._last_clamp_warn != (n_cuts, n_buckets):
            logger.warning(
                "Only %d cuts in this slice; clamping num_buckets %d → %d. "
                "If this is a curriculum phase, consider raising cl_schedule_len[0] "
                "so the model sees more variety than a single mini-batch repeated.",
                n_cuts, self.cfg.num_buckets, n_buckets,
            )
            self._last_clamp_warn = (n_cuts, n_buckets)
        return DynamicBucketingSampler(
            cuts,
            shuffle=shuffle,
            drop_last=drop_last,
            max_duration=self.cfg.max_duration,
            num_buckets=n_buckets,
        )

    def _make_loader(self, cuts, *, shuffle: bool, drop_last: bool) -> DataLoader:
        if self.tokenizer is None:
            raise RuntimeError("BPE model not configured.")
        # A slice too small to fill one max_duration batch would yield zero
        # batches under drop_last=True; keep the partial batch in that case.
        if drop_last and sum(c.duration for c in cuts) < self.cfg.max_duration:
            drop_last = False
        sampler = self._make_sampler(cuts, shuffle=shuffle, drop_last=drop_last)
        # Build a small wrapper Dataset that consults the sampler.
        dataset = CtcDataset(cuts, self.tokenizer)
        loader = DataLoader(
            dataset,
            batch_sampler=_LhotseSamplerAdapter(sampler, dataset),
            collate_fn=_collate_batch,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers if self.cfg.num_workers > 0 else False,
        )
        return loader

    # ----------------------------------------------------------------- API
    def train_loader(self, *, max_seconds_override: Optional[float] = None) -> DataLoader:
        """Build the training DataLoader.

        ``max_seconds_override`` is used by curriculum learning to cap utterance
        duration for a given phase; pass ``None`` to use ``data.max_seconds``.
        """
        all_train = None
        from lhotse import CutSet

        for part in self.cfg.train_parts:
            cuts = self._load_cuts(part, max_seconds_override=max_seconds_override)
            all_train = cuts if all_train is None else CutSet.from_cuts(list(all_train) + list(cuts))
        if all_train is None:
            raise RuntimeError("No train parts configured.")
        if len(all_train) == 0:
            cap = max_seconds_override if max_seconds_override is not None else self.cfg.max_seconds
            raise RuntimeError(
                f"No training cuts after filtering to [{self.cfg.min_seconds}, {cap}]s. "
                "Check data.min_seconds / data.max_seconds and curriculum.cl_schedule_len."
            )
        return self._make_loader(all_train, shuffle=self.cfg.shuffle, drop_last=self.cfg.drop_last)

    def valid_loader(self) -> DataLoader:
        cuts = self._load_cuts(self.cfg.dev_part)
        return self._make_loader(cuts, shuffle=False, drop_last=False)

    def test_loaders(self) -> dict[str, DataLoader]:
        out: dict[str, DataLoader] = {}
        for part in self.cfg.test_parts:
            cuts = self._load_cuts(part)
            out[part] = self._make_loader(cuts, shuffle=False, drop_last=False)
        return out


# ─────────────────────────────────────────────────────────────────────────
# Bridge between Lhotse's CutSampler and torch's batch_sampler protocol.
# ─────────────────────────────────────────────────────────────────────────


class _LhotseSamplerAdapter:
    """Yield lists of dataset indices instead of CutSets.

    Lhotse samplers yield ``CutSet`` objects; ``torch.utils.data.DataLoader``
    expects ``batch_sampler`` to yield lists of integer indices. This adapter
    maps cut IDs to dataset indices on the fly.
    """

    def __init__(self, sampler, dataset: CtcDataset) -> None:
        self.sampler = sampler
        self.dataset = dataset
        self._id_to_idx = {cid: i for i, cid in enumerate(dataset.ids)}

    def __iter__(self):
        for batch_cuts in self.sampler:
            yield [self._id_to_idx[c.id] for c in batch_cuts]

    def __len__(self):
        try:
            return len(self.sampler)
        except (AttributeError, TypeError):
            return 0
