"""BPE / unigram SentencePiece training.

Supports training multiple vocab sizes **in parallel** (one process per vocab)
so that ``--n_bpe 3000 300`` finishes ~as fast as the slowest single run.

The output is the standard SentencePiece pair: ``spm.model`` + ``spm.vocab``
under ``<bpe_root>/<dataset>_bpe<vocab>/``.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)


def default_bpe_path(data: str, vocab: int, *, root: str | os.PathLike = "downloads") -> Path:
    """Canonical location of a SentencePiece model for ``data`` × ``vocab``.

    All path construction goes through this helper so train/eval/error-message
    code stays in sync. The layout is ``<root>/bpe/<data>_bpe<vocab>/spm.model``,
    matching what :func:`train_bpe_parallel` writes.
    """
    return Path(root) / "bpe" / f"{data}_bpe{int(vocab)}" / "spm.model"


_OUTPUT_DIR_BPE_RE = re.compile(r"_\d+bpe(?=$|/)")


def rewrite_output_dir_bpe(output_dir: str, vocab: int) -> str:
    """Rewrite the ``_<N>bpe`` suffix of a config-supplied ``output_dir`` to match
    a user-overridden vocab size. Returns the input unchanged (with a warning) if
    no such suffix is present — caller should use ``--output-dir`` instead.
    """
    new, n = _OUTPUT_DIR_BPE_RE.subn(f"_{int(vocab)}bpe", output_dir, count=1)
    if n == 0:
        logger.warning(
            "output_dir=%r has no `_<N>bpe` suffix; --n_bpe=%d cannot be reflected "
            "in the output path. Pass --output-dir to override explicitly.",
            output_dir, int(vocab),
        )
    return new


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BpeTrainConfig:
    """SentencePiece training options.

    Defaults mirror IceFall's LibriSpeech recipe:
      - model_type = "unigram" (more robust than pure BPE for ASR)
      - character_coverage = 1.0 for English
      - no case normalization — case must match the runtime tokenizer input
        (see ``datamodule.py``, which feeds raw supervision text)
    """

    transcript_file: str = ""
    output_dir: str = ""
    vocab_size: int = 3000
    model_type: str = "unigram"          # "unigram" | "bpe" | "char"
    character_coverage: float = 1.0
    # SentencePiece auto-allocates <unk>/<s>/</s>/<pad>; do NOT list them here
    # or SentencePieceTrainer aborts with "must not be defined with
    # --control_symbols and --user_defined_symbols".
    user_defined_symbols: tuple[str, ...] = ()
    unk_id: int = 1
    bos_id: int = -1
    eos_id: int = -1
    pad_id: int = 0
    # ``<blank>`` is implicit in CTC; we keep ``pad_id=0`` (pad doubles as blank).
    treat_whitespace_as_suffix: bool = False
    input_sentence_size: int = 10_000_000
    shuffle_input_sentence: bool = True
    max_sentence_length: int = 16384
    num_threads: int = 4                 # per-vocab thread cap (multiplied by parallel vocabs)

    def model_prefix(self) -> str:
        return str(Path(self.output_dir) / "spm")


# ---------------------------------------------------------------------------
# Single-vocab training
# ---------------------------------------------------------------------------


def train_bpe(cfg: BpeTrainConfig, *, log: logging.Logger | None = None) -> Path:
    """Train one SentencePiece model. Returns path to ``spm.model``."""
    log = log or logger

    if not cfg.transcript_file or not Path(cfg.transcript_file).is_file():
        raise FileNotFoundError(f"transcript_file not found: {cfg.transcript_file}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    model_path = Path(cfg.model_prefix() + ".model")
    if model_path.exists():
        log.info(f"  [skip-bpe] {model_path} already exists.")
        return model_path

    # Lazy import — heavy.
    import sentencepiece as spm

    user_syms = ",".join(cfg.user_defined_symbols) if cfg.user_defined_symbols else ""

    log.info(
        f"  Training SentencePiece: vocab={cfg.vocab_size}, type={cfg.model_type}, "
        f"input={cfg.transcript_file}, out={cfg.output_dir}"
    )
    spm.SentencePieceTrainer.Train(
        input=cfg.transcript_file,
        model_prefix=cfg.model_prefix(),
        vocab_size=cfg.vocab_size,
        model_type=cfg.model_type,
        character_coverage=cfg.character_coverage,
        user_defined_symbols=user_syms,
        unk_id=cfg.unk_id,
        bos_id=cfg.bos_id,
        eos_id=cfg.eos_id,
        pad_id=cfg.pad_id,
        treat_whitespace_as_suffix=cfg.treat_whitespace_as_suffix,
        input_sentence_size=cfg.input_sentence_size,
        shuffle_input_sentence=cfg.shuffle_input_sentence,
        max_sentence_length=cfg.max_sentence_length,
        num_threads=cfg.num_threads,
    )

    log.info(f"  → {model_path}")
    return model_path


# ---------------------------------------------------------------------------
# Parallel multi-vocab training
# ---------------------------------------------------------------------------


def _worker(cfg: BpeTrainConfig) -> tuple[int, str]:
    """Child-process entry. Returns (vocab_size, model_path) on success."""
    # Reduce per-process log noise to the child.
    logging.basicConfig(level=logging.INFO, format="[bpe-%(process)d] %(message)s")
    path = train_bpe(cfg)
    return (cfg.vocab_size, str(path))


def train_bpe_parallel(
    *,
    transcript_file: str | Path,
    output_root: str | Path,
    vocab_sizes: Sequence[int],
    dataset_name: str,
    num_workers: int = 8,
    base_config: BpeTrainConfig | None = None,
    log: logging.Logger | None = None,
) -> dict[int, Path]:
    """Train multiple SentencePiece models in parallel.

    Args:
        transcript_file: One-utterance-per-line text file (gzipped *not* supported here —
            decompress upstream).
        output_root: Each vocab gets its own subdir ``<output_root>/<dataset>_bpe<V>/``.
        vocab_sizes: e.g. ``[3000, 300]``.
        dataset_name: Used in the output subdir name (e.g. ``"libri960"``).
        num_workers: Max parallel processes — capped at ``len(vocab_sizes)``.
        base_config: Defaults used for each vocab; only ``vocab_size`` /
            ``transcript_file`` / ``output_dir`` are overridden per-job.

    Returns: ``{vocab_size: Path(spm.model)}``.
    """
    log = log or logger
    transcript_file = str(Path(transcript_file).resolve())
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    base = base_config or BpeTrainConfig()

    # Per-job num_threads must respect the global num_workers budget so total
    # CPU threads don't explode: if we run 4 vocabs in parallel with 8 workers
    # total, each gets max(1, 8 // 4) = 2 threads.
    parallel_jobs = max(1, min(int(num_workers), len(vocab_sizes)))
    per_job_threads = max(1, int(num_workers) // parallel_jobs)

    jobs: list[BpeTrainConfig] = []
    for V in vocab_sizes:
        cfg = BpeTrainConfig(
            transcript_file=transcript_file,
            output_dir=str(output_root / f"{dataset_name}_bpe{V}"),
            vocab_size=int(V),
            model_type=base.model_type,
            character_coverage=base.character_coverage,
            user_defined_symbols=base.user_defined_symbols,
            unk_id=base.unk_id,
            bos_id=base.bos_id,
            eos_id=base.eos_id,
            pad_id=base.pad_id,
            treat_whitespace_as_suffix=base.treat_whitespace_as_suffix,
            input_sentence_size=base.input_sentence_size,
            shuffle_input_sentence=base.shuffle_input_sentence,
            max_sentence_length=base.max_sentence_length,
            num_threads=per_job_threads,
        )
        jobs.append(cfg)

    log.info(
        f"Training {len(jobs)} SentencePiece models in parallel "
        f"(parallel_jobs={parallel_jobs}, per_job_threads={per_job_threads})."
    )

    results: dict[int, Path] = {}
    if parallel_jobs == 1:
        for j in jobs:
            v, m = _worker(j)
            results[v] = Path(m)
    else:
        # Spawn context — safe with torch/transformers parents loaded.
        ctx = mp.get_context("spawn")
        with ctx.Pool(parallel_jobs) as pool:
            for v, m in pool.imap_unordered(_worker, jobs):
                results[v] = Path(m)

    return results
