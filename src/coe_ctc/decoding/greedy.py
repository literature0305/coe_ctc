"""CTC greedy decoding (argmax + collapse-blank).

Re-exports the implementation from ``coe_ctc.training.validation`` so
``coe_ctc.decoding.greedy`` is the canonical import path used by the
evaluation script.
"""

from __future__ import annotations

from typing import List

import torch

from coe_ctc.training.validation import ctc_greedy_decode

__all__ = ["ctc_greedy_decode", "greedy_decode_to_text"]


def greedy_decode_to_text(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    tokenizer,
    *,
    blank_idx: int = 0,
) -> List[str]:
    """Greedy decode → token-id sequences → text via SentencePiece."""
    decoded_ids = ctc_greedy_decode(log_probs, lengths, blank_idx=blank_idx)
    return [tokenizer.decode(ids) if ids else "" for ids in decoded_ids]
