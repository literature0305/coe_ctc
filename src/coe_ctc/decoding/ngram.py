"""Helpers for N-gram (KenLM) integration.

Two responsibilities:
  * Load an ARPA / binary KenLM model from disk.
  * Build a ``LanguageModel`` object that pyctcdecode can consume.

For richer rescoring (e.g. shallow fusion at training time), see the
``LMScorer`` class — a simple PyTorch wrapper that exposes ``score(token_ids)``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_kenlm_model(path: str | os.PathLike):
    """Return a kenlm.Model. Raises ImportError if kenlm is not installed."""
    try:
        import kenlm  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "kenlm is required for N-gram rescoring. Install with:\n"
            "    pip install https://github.com/kpu/kenlm/archive/master.zip"
        ) from exc

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"KenLM model not found: {path}")
    logger.info(f"Loading KenLM model: {p}")
    return kenlm.Model(str(p))


class LMScorer:
    """Thin wrapper that scores BPE-piece strings under a KenLM model.

    pyctcdecode itself wires the LM internally — this class is for callers
    that want manual rescoring (e.g. N-best post-processing in the future).
    """

    def __init__(self, model_path: str | os.PathLike) -> None:
        self.model = load_kenlm_model(model_path)

    def score(self, text: str) -> float:
        """Return log10-probability of *text* under the LM (KenLM convention)."""
        return float(self.model.score(text, bos=True, eos=True))

    def perplexity(self, text: str) -> float:
        return float(self.model.perplexity(text))
