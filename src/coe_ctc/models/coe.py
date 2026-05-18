"""Chain-of-Encoders (CoE) wrapper.

Stacks ``M`` virtual passes of a single weight-shared encoder. Each pass m
(0-based) sees the raw fBank features after a *cumulative* time-mask schedule
(pass 0 = most-masked, pass M-1 = least-masked) plus a single shared
frequency mask, and — for m ≥ 1 — attends to the previous pass's selected
layer output via per-layer K/V concat.

Configuration keys (parsed from the ``coe:`` section of the training YAML):

  * num_enc_chains         — M (default 5)
  * time_mask_ratios       — list[float] length M (default [0.1]*M)
  * loss_alphas            — list[float] or None; if None, derive
                               alpha_M=1, alpha_{m-1}=alpha_m*0.5,
                               then normalize to sum=1
  * pre_enc_layer_idx      — int (0-based, negative allowed); default = -1
                               (Zipformer ignores this — multi-rate)
  * pre_enc_feature_detach — default False
  * head_share             — default False (M independent CTC heads)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from coe_ctc.data.transforms import SpecAugment, SpecAugmentConfig
from coe_ctc.models.attention import _make_key_padding_mask
from coe_ctc.models.ctc import CTCHead, CTCOutput


@dataclass
class CoeOutput:
    """Aggregate output of a CoE forward pass.

    Mirrors ``CTCOutput`` for the *last* (least-masked) pass so that existing
    training/validation code paths that read ``loss``/``encoded_lens``/
    ``log_probs`` keep working. Per-pass details live in ``per_pass``.
    """

    log_probs: torch.Tensor
    encoded: torch.Tensor
    encoded_lens: torch.Tensor
    loss: Optional[torch.Tensor]
    per_pass: List[CTCOutput] = field(default_factory=list)
    alphas: List[float] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────
# Masking
# ─────────────────────────────────────────────────────────────────────────


class CoeMasking(nn.Module):
    """Produce the M progressively masked inputs for a CoE forward.

    Returns ``(freq_masked, time_masks)`` so the caller can apply the per-pass
    time mask lazily — avoiding M concurrent ``(B, T, F)`` fBank copies.

    In ``eval`` mode the time_masks tensor is ``None`` and ``freq_masked`` is
    the unmodified input (no masking at inference).
    """

    def __init__(
        self,
        *,
        num_chains: int,
        time_mask_ratios: Sequence[float],
        spec_aug_cfg: SpecAugmentConfig,
    ) -> None:
        super().__init__()
        if len(time_mask_ratios) != num_chains:
            raise ValueError(
                f"time_mask_ratios length ({len(time_mask_ratios)}) must equal "
                f"num_chains ({num_chains})."
            )
        self.num_chains = int(num_chains)
        self.time_mask_ratios = [float(r) for r in time_mask_ratios]
        # Reuse SpecAugment for freq masking (set num_time_masks=0 — CoE owns
        # the time-mask schedule). apply_prob=1 because training-mode gating
        # is already done in our own forward() before we call it.
        self._freq_aug = SpecAugment(
            SpecAugmentConfig(
                num_freq_masks=spec_aug_cfg.num_freq_masks,
                freq_mask_param=spec_aug_cfg.freq_mask_param,
                num_time_masks=0,
                time_mask_param=0,
                time_mask_ratio=0.0,
                apply_prob=1.0,
            )
        )

    def _build_cumulative_time_masks(
        self,
        batch_size: int,
        max_t: int,
        lengths: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Returns ``(M, B, T_raw)`` bool tensor; True at masked positions."""
        M = self.num_chains
        lengths_f = lengths.to(dtype=torch.float32, device=device)
        idx = torch.arange(max_t, device=device).unsqueeze(0)

        overlays = torch.zeros(M, batch_size, max_t, dtype=torch.bool, device=device)
        for i in range(M):
            ratio = self.time_mask_ratios[i]
            max_w = (lengths_f * ratio).floor().clamp(min=0).to(dtype=torch.long)
            rand_w = torch.rand(batch_size, device=device)
            widths = (rand_w * (max_w.to(dtype=torch.float32) + 1.0)).floor().to(dtype=torch.long)
            widths = widths.clamp(max=max_w)
            avail = (lengths - widths).clamp(min=0)
            rand_s = torch.rand(batch_size, device=device)
            starts = (rand_s * (avail.to(dtype=torch.float32) + 1.0)).floor().to(dtype=torch.long)
            starts = starts.clamp(max=avail)
            lo = starts.unsqueeze(1)
            hi = (starts + widths).unsqueeze(1)
            overlays[i] = (idx >= lo) & (idx < hi)

        # encoder m_enc (0-based) consumes the union over overlays[m_enc:M].
        encoder_masks = torch.zeros_like(overlays)
        encoder_masks[M - 1] = overlays[M - 1]
        for m_enc in range(M - 2, -1, -1):
            encoder_masks[m_enc] = encoder_masks[m_enc + 1] | overlays[m_enc]
        return encoder_masks

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.training:
            return features, None
        if features.dim() != 3:
            raise ValueError(f"CoeMasking expects (B,T,F); got {tuple(features.shape)}.")
        freq_masked = self._freq_aug(features, lengths)
        time_masks = self._build_cumulative_time_masks(
            features.size(0), features.size(1), lengths.to(features.device), features.device
        )
        return freq_masked, time_masks


# ─────────────────────────────────────────────────────────────────────────
# CoE model
# ─────────────────────────────────────────────────────────────────────────


def _normalize_alphas(alphas: Optional[Sequence[float]], M: int) -> List[float]:
    if alphas is None:
        raw = [2.0 ** (i - (M - 1)) for i in range(M)]
    else:
        raw = [float(a) for a in alphas]
        if len(raw) != M:
            raise ValueError(f"loss_alphas length ({len(raw)}) must equal M ({M}).")
    total = sum(raw)
    if total <= 0:
        raise ValueError(f"loss_alphas must sum to a positive value; got {raw}.")
    return [a / total for a in raw]


class CoeModel(nn.Module):
    """Chain-of-Encoders CTC model.

    Holds **one** encoder (the same parameters are used M times) and either a
    single CTC head (``head_share=True``) or M heads (``head_share=False``).
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        vocab_size: int,
        num_enc_chains: int = 5,
        time_mask_ratios: Sequence[float] = (0.1, 0.1, 0.1, 0.1, 0.1),
        loss_alphas: Optional[Sequence[float]] = None,
        pre_enc_layer_idx: int = -1,
        pre_enc_feature_detach: bool = False,
        head_share: bool = False,
        spec_aug_cfg: Optional[SpecAugmentConfig] = None,
        blank_idx: int = 0,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not hasattr(encoder, "d_model"):
            raise AttributeError("Encoder must expose a `d_model` attribute.")
        if not hasattr(encoder, "num_layers"):
            raise AttributeError("Encoder must expose a `num_layers` property.")
        if num_enc_chains < 1:
            raise ValueError(f"num_enc_chains must be ≥ 1; got {num_enc_chains}.")

        self.encoder = encoder
        self.num_enc_chains = int(num_enc_chains)
        self.blank_idx = int(blank_idx)
        self.vocab_size = int(vocab_size)
        self.head_share = bool(head_share)
        self.pre_enc_feature_detach = bool(pre_enc_feature_detach)

        n_layers = encoder.num_layers
        resolved_idx = int(pre_enc_layer_idx)
        if resolved_idx < 0:
            resolved_idx = n_layers + resolved_idx
        if not (0 <= resolved_idx < n_layers):
            raise ValueError(
                f"pre_enc_layer_idx={pre_enc_layer_idx} out of range "
                f"for encoder with {n_layers} layers."
            )
        self.pre_enc_layer_idx = resolved_idx

        self.alphas = _normalize_alphas(loss_alphas, num_enc_chains)

        n_heads = 1 if head_share else num_enc_chains
        self.heads = nn.ModuleList(
            [CTCHead(encoder.d_model, vocab_size, dropout=head_dropout) for _ in range(n_heads)]
        )

        self.masking = CoeMasking(
            num_chains=num_enc_chains,
            time_mask_ratios=time_mask_ratios,
            spec_aug_cfg=spec_aug_cfg or SpecAugmentConfig(),
        )

    @torch.no_grad()
    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def parameter_breakdown(self) -> dict:
        out: dict = {}
        for name, m in self.named_children():
            out[name] = sum(p.numel() for p in m.parameters())
        return out

    def _head_for(self, m: int) -> CTCHead:
        return self.heads[0] if self.head_share else self.heads[m]

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        *,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> CoeOutput:
        freq_masked, time_masks = self.masking(features, feature_lengths)

        per_pass: List[CTCOutput] = []
        losses: List[torch.Tensor] = []
        pre_enc: Optional[torch.Tensor] = None
        pre_enc_pad: Optional[torch.Tensor] = None

        for m in range(self.num_enc_chains):
            if time_masks is None:
                x_m = freq_masked
            else:
                x_m = freq_masked.masked_fill(time_masks[m].unsqueeze(-1), 0.0)

            enc = self.encoder(
                x_m,
                feature_lengths,
                pre_enc=pre_enc,
                pre_enc_padding_mask=pre_enc_pad,
                return_layer=self.pre_enc_layer_idx,
            )
            log_probs = self._head_for(m)(enc.encoded)

            loss = None
            if targets is not None:
                if target_lengths is None:
                    raise ValueError("target_lengths must be provided when targets is given.")
                loss = F.ctc_loss(
                    log_probs=log_probs.transpose(0, 1),
                    targets=targets,
                    input_lengths=enc.encoded_lens.to(dtype=torch.long),
                    target_lengths=target_lengths.to(dtype=torch.long),
                    blank=self.blank_idx,
                    reduction="mean",
                    zero_infinity=True,
                )
                losses.append(loss)

            per_pass.append(
                CTCOutput(
                    log_probs=log_probs,
                    encoded=enc.encoded,
                    encoded_lens=enc.encoded_lens,
                    loss=loss,
                )
            )

            if m < self.num_enc_chains - 1:
                next_pre = enc.selected
                if self.pre_enc_feature_detach:
                    next_pre = next_pre.detach()
                pre_enc = next_pre
                pre_enc_pad = _make_key_padding_mask(enc.encoded_lens, max_len=pre_enc.size(1))

        total_loss: Optional[torch.Tensor] = None
        if losses:
            total_loss = sum(a * l for a, l in zip(self.alphas, losses))

        last = per_pass[-1]
        return CoeOutput(
            log_probs=last.log_probs,
            encoded=last.encoded,
            encoded_lens=last.encoded_lens,
            loss=total_loss,
            per_pass=per_pass,
            alphas=list(self.alphas),
        )
