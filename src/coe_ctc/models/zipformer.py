"""Zipformer encoder — multi-rate downsampled Conformer stacks.

This is a *simplified* reimplementation of icefall's Zipformer (Yao et al.,
2023). The key idea: instead of one stack at the subsampled rate, run
*several* stacks at progressively coarser temporal resolutions, then upsample
+ blend the outputs back. The original paper also introduces ``BiasNorm``,
``ScaledAdam`` and ``Eden`` LR schedules; here we keep ``LayerNorm`` + AdamW
to stay framework-agnostic. Performance is competitive but a few percent
short of full Zipformer-Large for the very largest configs.

The implementation follows the same I/O contract as ``ConformerEncoder``:
    x      (B, T, num_features)
    x_lens (B,)
    →
    out    (B, T', d_model)
    out_lens (B,)
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from coe_ctc.models.attention import (
    EncoderOutput,
    RelPositionalEncoding,
    RelPosMultiHeadAttention,
    _make_key_padding_mask,
)
from coe_ctc.models.conformer import ConformerFeedForward, ConvModule
from coe_ctc.models.subsampling import Conv2dSubsampling


# ─────────────────────────────────────────────────────────────────────────
# COE helper: resample pre_enc feature to a target time length per stack
# ─────────────────────────────────────────────────────────────────────────


def _pool_pre_enc(
    pre_enc: Optional[torch.Tensor],
    pre_enc_padding_mask: Optional[torch.Tensor],
    target_len: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Adaptive-pool ``pre_enc`` (B,T,D) to ``target_len`` along time.

    The padding mask is max-pooled (any pad in the pool window → pad out).
    Returns ``(None, None)`` when ``pre_enc`` itself is ``None`` — used by
    Zipformer to keep call sites concise.
    """
    if pre_enc is None:
        return None, None
    if pre_enc.size(1) == target_len:
        return pre_enc, pre_enc_padding_mask
    pooled = F.adaptive_avg_pool1d(pre_enc.transpose(1, 2), target_len).transpose(1, 2)
    pooled_pad: Optional[torch.Tensor] = None
    if pre_enc_padding_mask is not None:
        pad_float = pre_enc_padding_mask.to(dtype=pre_enc.dtype).unsqueeze(1)
        pad_pooled = F.adaptive_max_pool1d(pad_float, target_len).squeeze(1) > 0.5
        pooled_pad = pad_pooled
    return pooled, pooled_pad


# ─────────────────────────────────────────────────────────────────────────
# Per-stack Zipformer block (same Macaron FFN/Attn/Conv/FFN structure)
# ─────────────────────────────────────────────────────────────────────────


class ZipformerBlock(nn.Module):
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
        x = x + 0.5 * self.ffn1(x)
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
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.norm_final(x)


# ─────────────────────────────────────────────────────────────────────────
# Downsample / upsample blocks (factor 2 each)
# ─────────────────────────────────────────────────────────────────────────


class Downsample(nn.Module):
    """Average-pool by ``factor`` along time, then a per-frame linear mix.

    The length update is ``L // factor`` to match
    ``F.avg_pool1d(kernel_size=factor, stride=factor, ceil_mode=False)`` —
    further clamped to the actual pooled tensor size so we never advertise
    frames the tensor doesn't have.
    """

    def __init__(self, d_model: int, factor: int) -> None:
        super().__init__()
        self.factor = factor
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.factor <= 1:
            return self.proj(x), x_lens
        x_pooled = F.avg_pool1d(
            x.transpose(1, 2),
            kernel_size=self.factor,
            stride=self.factor,
            ceil_mode=False,
        ).transpose(1, 2)
        x_out = self.proj(x_pooled)
        new_lens = (x_lens // self.factor).clamp(max=x_out.size(1))
        return x_out, new_lens


class Upsample(nn.Module):
    """Nearest-neighbour upsample to an exact ``target_len`` along time.

    We use ``F.interpolate`` rather than ``repeat_interleave`` so we can
    produce any output length — the latter would round to multiples of
    ``factor`` and trip an off-by-one when ``base_len % factor != 0``.
    """

    def __init__(self, d_model: int, factor: int) -> None:
        super().__init__()
        self.factor = factor
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        if self.factor <= 1:
            x = self.proj(x)
            return x[:, :target_len]
        x = F.interpolate(x.transpose(1, 2), size=target_len, mode="nearest").transpose(1, 2)
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────
# Top-level encoder
# ─────────────────────────────────────────────────────────────────────────


class ZipformerEncoder(nn.Module):
    """Multi-rate Zipformer encoder.

    ``downsampling_factors`` and ``num_layers_per_stack`` must have the same
    length. The first stack runs at the post-subsampling rate (1×); each
    subsequent stack downsamples by its factor (cumulative), then everything
    is upsampled back and **summed** with the highest-resolution residual.
    """

    def __init__(
        self,
        *,
        num_features: int = 80,
        d_model: int = 256,
        num_heads: int = 4,
        d_ff: int = 1024,
        kernel_size: int = 31,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        subsampling_dropout: float = 0.1,
        downsampling_factors: Sequence[int] = (1, 2, 4, 2, 1),
        num_layers_per_stack: Sequence[int] = (2, 4, 6, 4, 2),
    ) -> None:
        super().__init__()
        if len(downsampling_factors) != len(num_layers_per_stack):
            raise ValueError("downsampling_factors and num_layers_per_stack must match in length.")

        self.num_features = num_features
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.kernel_size = kernel_size
        self.dropout_p = dropout
        self.attn_dropout_p = attn_dropout
        self.downsampling_factors = tuple(downsampling_factors)
        self.num_layers_per_stack = tuple(num_layers_per_stack)
        self.subsampling = Conv2dSubsampling(num_features, d_model, dropout=subsampling_dropout)
        self.rel_pos = RelPositionalEncoding(d_model, dropout=dropout)

        # Each stack runs at its OWN downsample rate relative to the base.
        # Downsamplers are absolute (base → stack rate); upsamplers are absolute
        # (stack rate → base). Sequences like (1, 2, 4, 2, 1) are valid because
        # we feed every stack fresh from the base output instead of chaining.
        self.stacks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for factor, n_layers in zip(downsampling_factors, num_layers_per_stack):
            self.downsamplers.append(Downsample(d_model, factor))
            self.upsamplers.append(Upsample(d_model, factor))
            self.stacks.append(
                nn.ModuleList(
                    [
                        ZipformerBlock(
                            d_model,
                            num_heads,
                            d_ff,
                            kernel_size=kernel_size,
                            dropout=dropout,
                            attn_dropout=attn_dropout,
                        )
                        for _ in range(n_layers)
                    ]
                )
            )
        self.final_norm = nn.LayerNorm(d_model)

    @property
    def num_layers(self) -> int:
        return sum(len(s) for s in self.stacks)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        pre_enc: Optional[torch.Tensor] = None,
        pre_enc_padding_mask: Optional[torch.Tensor] = None,
        return_layer: Optional[int] = None,
    ) -> EncoderOutput:
        # ``return_layer`` is ignored: intermediate stacks live at varying
        # time rates, so only the final post-norm output (base rate) is a
        # well-defined pre_enc_feature for the next CoE chain step.
        x, x_lens = self.subsampling(x, x_lens)
        base_x, _ = self.rel_pos(x)
        target_len = base_x.size(1)

        # Stacks with the same downsampling factor (e.g. (1,2,4,2,1) → ×2
        # twice) would otherwise re-pool pre_enc identically; cache by len.
        pool_cache: dict[int, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {}

        out = torch.zeros_like(base_x)
        for stack, down, up in zip(self.stacks, self.downsamplers, self.upsamplers):
            cur_x, cur_lens = down(base_x, x_lens)
            _, pos_emb = self.rel_pos(cur_x)
            kpm = _make_key_padding_mask(cur_lens, max_len=cur_x.size(1))
            tlen = cur_x.size(1)
            if pre_enc is None:
                stack_pre_enc, stack_pre_pad = None, None
            else:
                if tlen not in pool_cache:
                    pool_cache[tlen] = _pool_pre_enc(pre_enc, pre_enc_padding_mask, tlen)
                stack_pre_enc, stack_pre_pad = pool_cache[tlen]
            for block in stack:
                cur_x = block(
                    cur_x,
                    pos_emb=pos_emb,
                    key_padding_mask=kpm,
                    pre_enc=stack_pre_enc,
                    pre_enc_padding_mask=stack_pre_pad,
                )
            out = out + up(cur_x, target_len=target_len)
        out = self.final_norm(out)
        return EncoderOutput(out, x_lens, out)
