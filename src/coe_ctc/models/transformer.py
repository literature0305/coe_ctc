"""Plain Transformer encoder (pre-norm, GELU, dropout) for CTC ASR.

Used as the simplest baseline encoder. See ``conformer.py`` for the
better-performing Conformer encoder, and ``zipformer.py`` for the
multi-rate Zipformer.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from coe_ctc.models.attention import (
    EncoderOutput,
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    _make_key_padding_mask,
)
from coe_ctc.models.subsampling import Conv2dSubsampling


class FeedForward(nn.Module):
    """Two-layer FFN with GELU activation (Transformer style)."""

    def __init__(self, d_model: int, d_ff: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    """Pre-LayerNorm Transformer encoder block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=attn_dropout)
        self.dropout_attn = nn.Dropout(dropout)

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        pre_enc: Optional[torch.Tensor] = None,
        pre_enc_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm + residual
        residual = x
        x = self.norm_attn(x)
        x = self.attn(
            x,
            key_padding_mask=key_padding_mask,
            pre_enc=pre_enc,
            pre_enc_padding_mask=pre_enc_padding_mask,
        )
        x = residual + self.dropout_attn(x)

        residual = x
        x = self.norm_ffn(x)
        x = residual + self.ffn(x)
        return x


class TransformerEncoder(nn.Module):
    """fBank → Conv subsampling → N transformer blocks → (B, T', d_model).

    The final ``LayerNorm`` is applied once after the stack to match the
    Pre-Norm convention.
    """

    def __init__(
        self,
        *,
        num_features: int = 80,
        d_model: int = 256,
        num_layers: int = 12,
        num_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        subsampling_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_p = dropout
        self.attn_dropout_p = attn_dropout
        self.subsampling = Conv2dSubsampling(num_features, d_model, dropout=subsampling_dropout)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, num_heads, d_ff, dropout=dropout, attn_dropout=attn_dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        pre_enc: Optional[torch.Tensor] = None,
        pre_enc_padding_mask: Optional[torch.Tensor] = None,
        return_layer: Optional[int] = None,
    ) -> EncoderOutput:
        x, x_lens = self.subsampling(x, x_lens)
        x = self.pos_enc(x)
        max_len = x.size(1)
        key_padding_mask = _make_key_padding_mask(x_lens, max_len=max_len)
        selected: Optional[torch.Tensor] = None
        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                key_padding_mask=key_padding_mask,
                pre_enc=pre_enc,
                pre_enc_padding_mask=pre_enc_padding_mask,
            )
            if return_layer is not None and i == return_layer:
                selected = x
        x = self.final_norm(x)
        if selected is None:
            selected = x
        return EncoderOutput(x, x_lens, selected)
