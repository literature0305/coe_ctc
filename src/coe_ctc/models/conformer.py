"""Conformer encoder — Macaron FFN + MHSA(rel-pos) + ConvModule + FFN.

Reference: Gulati et al., "Conformer: Convolution-augmented Transformer for
Speech Recognition" (2020). Implementation follows IceFall / ESPNet defaults:
kernel size 31, GLU + BatchNorm + Swish in the conv module.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from coe_ctc.models.attention import (
    EncoderOutput,
    RelPosMultiHeadAttention,
    RelPositionalEncoding,
    _make_key_padding_mask,
)
from coe_ctc.models.subsampling import Conv2dSubsampling


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x * torch.sigmoid(x)


class ConformerFeedForward(nn.Module):
    """Macaron-style FFN (Swish, 2× expansion typical), half-residual."""

    def __init__(self, d_model: int, d_ff: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvModule(nn.Module):
    """Conformer convolution module.

    LayerNorm → PointwiseConv (×2 channels) → GLU → DepthwiseConv → BatchNorm
    → Swish → PointwiseConv → Dropout.
    """

    def __init__(self, d_model: int, *, kernel_size: int = 31, dropout: float = 0.1) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("ConvModule kernel size must be odd.")
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.act = Swish()
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = self.norm(x)
        x = x.transpose(1, 2)                        # (B, D, T)
        x = self.pointwise_in(x)                     # (B, 2D, T)
        x = F.glu(x, dim=1)                          # (B, D, T)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.pointwise_out(x)
        x = x.transpose(1, 2)                        # (B, T, D)
        return self.dropout(x)


class ConformerBlock(nn.Module):
    """Single Conformer encoder block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        kernel_size: int = 31,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model, d_ff, dropout=dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = RelPosMultiHeadAttention(d_model, num_heads, dropout=attn_dropout)
        self.dropout_attn = nn.Dropout(dropout)
        self.conv = ConvModule(d_model, kernel_size=kernel_size, dropout=dropout)
        self.ffn2 = ConformerFeedForward(d_model, d_ff, dropout=dropout)
        self.norm_final = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        pre_enc: Optional[torch.Tensor] = None,
        pre_enc_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # FFN ½ residual
        x = x + 0.5 * self.ffn1(x)
        # MHSA with rel-pos
        residual = x
        x = self.norm_attn(x)
        x = self.attn(
            x,
            pos_emb=pos_emb,
            key_padding_mask=key_padding_mask,
            pre_enc=pre_enc,
            pre_enc_padding_mask=pre_enc_padding_mask,
        )
        x = residual + self.dropout_attn(x)
        # Conv module
        x = x + self.conv(x)
        # FFN ½ residual
        x = x + 0.5 * self.ffn2(x)
        return self.norm_final(x)


class ConformerEncoder(nn.Module):
    """fBank → Conv subsampling → N Conformer blocks → (B, T', d_model)."""

    def __init__(
        self,
        *,
        num_features: int = 80,
        d_model: int = 256,
        num_layers: int = 12,
        num_heads: int = 4,
        d_ff: int = 1024,
        kernel_size: int = 31,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        subsampling_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.subsampling = Conv2dSubsampling(num_features, d_model, dropout=subsampling_dropout)
        self.rel_pos = RelPositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )

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
        x, pos_emb = self.rel_pos(x)
        max_len = x.size(1)
        key_padding_mask = _make_key_padding_mask(x_lens, max_len=max_len)
        selected: Optional[torch.Tensor] = None
        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                pos_emb=pos_emb,
                key_padding_mask=key_padding_mask,
                pre_enc=pre_enc,
                pre_enc_padding_mask=pre_enc_padding_mask,
            )
            if return_layer is not None and i == return_layer:
                selected = x
        if selected is None:
            selected = x
        return EncoderOutput(x, x_lens, selected)
