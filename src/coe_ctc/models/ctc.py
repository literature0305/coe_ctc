"""CTC head + the full encoder-only CTC model wrapper.

The model is intentionally minimal: encoder + linear projection + log-softmax,
then ``F.ctc_loss`` for training. Decoding is delegated to
``coe_ctc.decoding`` (Phase 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CTCOutput:
    """Convenience container returned by ``CtcModel.forward``."""

    log_probs: torch.Tensor   # (B, T', V) log-softmax over vocab
    encoded: torch.Tensor     # (B, T', d_model) raw encoder output
    encoded_lens: torch.Tensor  # (B,)
    loss: Optional[torch.Tensor] = None  # set if targets provided


class CTCHead(nn.Module):
    """Linear projection from encoder dim → vocab size + log-softmax."""

    def __init__(self, d_model: int, vocab_size: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.proj(self.dropout(x)), dim=-1)


class CtcModel(nn.Module):
    """Encoder + CTC head.

    The encoder must accept ``(x, x_lens)`` and return ``(encoded, encoded_lens)``
    — both ``TransformerEncoder``, ``ConformerEncoder`` and ``ZipformerEncoder``
    in this package satisfy this contract.

    The blank index is fixed to ``0`` to match our SentencePiece convention
    (``pad_id=0`` doubles as the CTC blank label). Override with
    ``blank_idx`` if you re-train BPE with a different special-token layout.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        vocab_size: int,
        blank_idx: int = 0,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not hasattr(encoder, "d_model"):
            raise AttributeError("Encoder must expose a `d_model` attribute.")
        self.encoder = encoder
        self.head = CTCHead(encoder.d_model, vocab_size, dropout=head_dropout)
        self.blank_idx = blank_idx
        self.vocab_size = vocab_size

    # ----------------------------------------------------------------- utils
    @torch.no_grad()
    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def parameter_breakdown(self) -> dict[str, int]:
        """Return params-per-submodule for the health report's Model section."""
        out: dict[str, int] = {}
        for name, m in self.named_children():
            out[name] = sum(p.numel() for p in m.parameters())
        return out

    # ----------------------------------------------------------------- core
    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        *,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> CTCOutput:
        """
        Args:
            features:        (B, T, num_features) e.g. fBank-80.
            feature_lengths: (B,) frame counts before subsampling.
            targets:         (B, U) padded token ids (no blanks).
            target_lengths:  (B,) actual token counts (U_i ≤ U).

        Returns:
            CTCOutput. ``loss`` is populated iff ``targets`` is provided.
        """
        enc_out = self.encoder(features, feature_lengths)
        encoded, encoded_lens = enc_out.encoded, enc_out.encoded_lens
        log_probs = self.head(encoded)  # (B, T', V)

        loss = None
        if targets is not None:
            if target_lengths is None:
                raise ValueError("target_lengths must be provided when targets is given.")
            # F.ctc_loss expects (T, B, V) log-probs.
            loss = F.ctc_loss(
                log_probs=log_probs.transpose(0, 1),
                targets=targets,
                input_lengths=encoded_lens.to(dtype=torch.long),
                target_lengths=target_lengths.to(dtype=torch.long),
                blank=self.blank_idx,
                reduction="mean",
                zero_infinity=True,
            )

        return CTCOutput(log_probs=log_probs, encoded=encoded, encoded_lens=encoded_lens, loss=loss)
