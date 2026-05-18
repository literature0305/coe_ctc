"""Multi-head attention with Flash/SDPA backend.

We use ``torch.nn.functional.scaled_dot_product_attention`` which dispatches
to FlashAttention 2 on Ampere+ GPUs, memory-efficient attention on Turing,
and a math kernel on CPU. No manual softmax / matmul — that's the only way
to consistently hit SOTA throughput on A100/H100.

Two flavors:
  * ``MultiHeadAttention``       — plain self-attention (Transformer encoder).
  * ``RelPosMultiHeadAttention`` — Shaw-style relative positions used by
                                    Conformer. We follow ESPNet's relative
                                    positional encoding rather than the more
                                    complex Transformer-XL variant.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderOutput(NamedTuple):
    """Shared 3-tuple return shape for all encoders.

    ``selected`` is the layer output picked for the next CoE chain step
    (defaults to the final encoder output when no specific layer was
    requested). Plain CTC training reads ``encoded``/``encoded_lens`` only.
    """

    encoded: torch.Tensor
    encoded_lens: torch.Tensor
    selected: torch.Tensor


def _make_key_padding_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Bool mask of shape (B, T) where True = pad position (to be ignored)."""
    if max_len is None:
        max_len = int(lengths.max().item())
    idx = torch.arange(max_len, device=lengths.device)
    return idx.unsqueeze(0) >= lengths.unsqueeze(1)


def _concat_padding_masks(
    curr_mask: Optional[torch.Tensor],
    pre_mask: Optional[torch.Tensor],
    b: int,
    t_curr: int,
    t_pre: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Combine the current and pre_enc key-padding masks for COE attention."""
    if curr_mask is None and pre_mask is None:
        return None
    if curr_mask is None:
        curr_mask = torch.zeros(b, t_curr, dtype=torch.bool, device=device)
    if pre_mask is None:
        pre_mask = torch.zeros(b, t_pre, dtype=torch.bool, device=device)
    return torch.cat([curr_mask, pre_mask], dim=1)


# ─────────────────────────────────────────────────────────────────────────
# Vanilla multi-head attention (used by the Transformer encoder)
# ─────────────────────────────────────────────────────────────────────────


class MultiHeadAttention(nn.Module):
    """Standard scaled-dot-product MHA with Flash/SDPA dispatch."""

    def __init__(self, d_model: int, num_heads: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        pre_enc: Optional[torch.Tensor] = None,
        pre_enc_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) input.
            key_padding_mask: (B, T) bool, True at pad positions.
            pre_enc: (B, T_pre, D) optional COE pre-encoder feature. When
                given, K/V are built from ``concat([x, pre_enc], dim=time)``.
            pre_enc_padding_mask: (B, T_pre) bool, True at pre_enc pad
                positions. Required iff pre_enc has padding.

        Returns:
            (B, T, D) output.
        """
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        if pre_enc is None:
            kv_input = x
            t_kv = t
            kv_mask = key_padding_mask
        else:
            t_pre = pre_enc.size(1)
            kv_input = torch.cat([x, pre_enc], dim=1)
            t_kv = t + t_pre
            kv_mask = _concat_padding_masks(key_padding_mask, pre_enc_padding_mask, b, t, t_pre, x.device)

        k = self.k_proj(kv_input).view(b, t_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_input).view(b, t_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if kv_mask is not None:
            attn_mask = torch.zeros(
                b, 1, 1, t_kv, dtype=q.dtype, device=q.device
            ).masked_fill(kv_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out_proj(out)


# ─────────────────────────────────────────────────────────────────────────
# Sinusoidal positional encoding (shared by Transformer + Conformer)
# ─────────────────────────────────────────────────────────────────────────


class SinusoidalPositionalEncoding(nn.Module):
    """Absolute positional encoding added to the input embedding."""

    def __init__(self, d_model: int, *, max_len: int = 10000, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────
# Relative positional encoding (Conformer)
# ─────────────────────────────────────────────────────────────────────────


class RelPositionalEncoding(nn.Module):
    """ESPNet-style symmetric relative positional encoding.

    Returns a (1, 2T-1, D) buffer to be projected & added inside attention.
    """

    def __init__(self, d_model: int, *, max_len: int = 5000, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self._extend(max_len)

    def _extend(self, max_len: int) -> None:
        pe_pos = torch.zeros(max_len, self.d_model)
        pe_neg = torch.zeros(max_len, self.d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float) * (-math.log(10000.0) / self.d_model))
        pe_pos[:, 0::2] = torch.sin(pos * div)
        pe_pos[:, 1::2] = torch.cos(pos * div)
        pe_neg[:, 0::2] = torch.sin(-pos * div)
        pe_neg[:, 1::2] = torch.cos(-pos * div)
        pe_pos = torch.flip(pe_pos, dims=[0]).unsqueeze(0)
        pe_neg = pe_neg[1:].unsqueeze(0)
        pe = torch.cat([pe_pos, pe_neg], dim=1)
        self.register_buffer("pe", pe, persistent=False)
        self._max_len = max_len

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_after_dropout, rel_pos_encoding)."""
        t = x.size(1)
        if t > self._max_len:
            self._extend(t * 2)
            self.pe = self.pe.to(device=x.device, dtype=x.dtype)
        center = (self.pe.size(1) + 1) // 2 - 1  # index of position 0
        pos_emb = self.pe[:, center - t + 1 : center + t]
        return self.dropout(x), pos_emb.to(dtype=x.dtype)


class RelPosMultiHeadAttention(nn.Module):
    """Conformer-style relative-position MHA.

    Implements the Shaw / Transformer-XL hybrid used in ESPNet — projects the
    relative-position embedding through a separate linear and folds the
    "skew" trick into attention scores. Falls back to SDPA for the standard
    content-content branch when there is no padding mask conflict.
    """

    def __init__(self, d_model: int, num_heads: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable u/v biases (Transformer-XL).
        self.bias_u = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.bias_v = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.bias_u.unsqueeze(0))
        nn.init.xavier_uniform_(self.bias_v.unsqueeze(0))

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        """Shift relative-position scores so column j ↔ rel-pos j-i.

        Input  (B, H, T, 2T-1)
        Output (B, H, T, T)
        """
        b, h, t, k = x.shape
        zero_pad = x.new_zeros(b, h, t, 1)
        x_padded = torch.cat([zero_pad, x], dim=-1)        # (B,H,T,2T)
        x_padded = x_padded.view(b, h, k + 1, t)            # reshape
        x = x_padded[:, :, 1:].view(b, h, t, k)             # drop first row
        return x[:, :, :, :t]

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        pre_enc: Optional[torch.Tensor] = None,
        pre_enc_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim)
        k_curr = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v_curr = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        if pre_enc is not None:
            t_pre = pre_enc.size(1)
            if t_pre != t:
                # RPE branch assumes K_pre time matches Q time so the same rel-pos
                # signal can be reused for both halves of K (see _rel_shift).
                raise ValueError(
                    f"RelPosMultiHeadAttention requires pre_enc time ({t_pre}) "
                    f"== current time ({t}); resample pre_enc upstream."
                )
            k_pre = self.k_proj(pre_enc).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
            v_pre = self.v_proj(pre_enc).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
            k = torch.cat([k_curr, k_pre], dim=2)            # (B, H, 2T, D_h)
            v = torch.cat([v_curr, v_pre], dim=2)
        else:
            k = k_curr
            v = v_curr

        # Add learnable biases to query before the two attention branches.
        q_with_u = (q + self.bias_u).transpose(1, 2)        # (B, H, T, D_h)
        q_with_v = (q + self.bias_v).transpose(1, 2)

        # AC: content-content over the full K (T or 2T).
        ac = torch.matmul(q_with_u, k.transpose(-2, -1))    # (B, H, T, T or 2T)

        # BD: content-pos. RPE is computed once over T-T and duplicated for
        # the pre_enc half — frame i ↔ frame i in both halves.
        p = self.pos_proj(pos_emb).view(pos_emb.size(0), pos_emb.size(1), self.num_heads, self.head_dim)
        p = p.permute(0, 2, 3, 1).squeeze(0)                # (H, D_h, 2T-1)
        bd_curr = torch.matmul(q_with_v, p)                 # (B, H, T, 2T-1)
        bd_curr = self._rel_shift(bd_curr)                  # (B, H, T, T)
        bd = torch.cat([bd_curr, bd_curr], dim=-1) if pre_enc is not None else bd_curr

        scores = (ac + bd) * self.scale
        kv_mask: Optional[torch.Tensor]
        if pre_enc is not None:
            kv_mask = _concat_padding_masks(key_padding_mask, pre_enc_padding_mask, b, t, t, x.device)
        else:
            kv_mask = key_padding_mask
        if kv_mask is not None:
            scores = scores.masked_fill(kv_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        if self.training and self.dropout_p > 0:
            attn = F.dropout(attn, p=self.dropout_p)
        out = torch.matmul(attn, v)                         # (B, H, T, D_h)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out_proj(out)
