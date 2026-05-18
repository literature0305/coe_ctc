"""Audio / feature-level transforms applied at training time.

The principal one is :class:`SpecAugment`, applied lazily on each batch
(operates on the **post-fbank** feature tensor, never on raw audio).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────
# SpecAugment
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class SpecAugmentConfig:
    """SpecAugment params (Park et al., 2019).

    Defaults match the IceFall "LD" policy used for LibriSpeech-960h, scaled
    down a notch so a 2080 Ti debug run doesn't blow up loss in the first
    1000 steps.
    """

    # Frequency masking
    num_freq_masks: int = 2
    freq_mask_param: int = 27       # max F to mask in [0, F]
    # Time masking
    num_time_masks: int = 10
    time_mask_param: int = 100      # max T to mask in [0, T]
    time_mask_ratio: float = 0.04   # cap: mask ≤ ratio × seq length per mask
    # Time warping (lightweight; full warp is expensive — we use a small subset)
    enable_time_warp: bool = False
    time_warp_param: int = 5
    # Probability of applying SpecAugment per batch (1.0 = always)
    apply_prob: float = 1.0


class SpecAugment(nn.Module):
    """In-place SpecAugment on a feature batch.

    Expects input of shape ``(B, T, F)`` with valid lengths in ``lengths``.
    Pad regions (positions ``>= lengths[i]``) are left untouched so masks can
    still cover real frames without leakage.
    """

    def __init__(self, cfg: SpecAugmentConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SpecAugmentConfig()

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.training:
            return x
        if torch.rand(1).item() > self.cfg.apply_prob:
            return x
        if x.dim() != 3:
            raise ValueError(f"SpecAugment expects (B,T,F); got {tuple(x.shape)}.")

        x = x.clone()  # we mutate; protect the upstream tensor
        b, t, f = x.shape
        device = x.device

        if lengths is None:
            lengths = torch.full((b,), t, device=device, dtype=torch.long)
        lengths = lengths.to(device=device, dtype=torch.long).clamp(max=t)

        # ── frequency masking — vectorized batch mask along the F axis ─────
        if self.cfg.num_freq_masks > 0 and self.cfg.freq_mask_param > 0:
            n_masks = self.cfg.num_freq_masks
            widths = torch.randint(0, self.cfg.freq_mask_param + 1, (n_masks, b), device=device)
            starts = (
                torch.rand(n_masks, b, device=device) * (f - widths).clamp(min=1).float()
            ).long()
            freq_idx = torch.arange(f, device=device).view(1, 1, f)
            mask = torch.zeros(b, 1, f, device=device, dtype=torch.bool)
            for m in range(n_masks):
                lo = starts[m].view(b, 1, 1)
                hi = (starts[m] + widths[m]).view(b, 1, 1)
                mask |= (freq_idx >= lo) & (freq_idx < hi)
            # broadcast over T
            x = x.masked_fill(mask.expand(-1, t, -1), 0.0)

        # ── time masking — per-sample (lengths vary) ───────────────────────
        max_t_mask = self.cfg.time_mask_param
        ratio = self.cfg.time_mask_ratio
        for _ in range(self.cfg.num_time_masks):
            for i in range(b):
                Li = int(lengths[i].item())
                if Li <= 1:
                    continue
                cap = min(max_t_mask, int(Li * ratio))
                if cap < 1:
                    continue
                mt = int(torch.randint(1, cap + 1, (1,)).item())
                t0 = int(torch.randint(0, max(1, Li - mt + 1), (1,)).item())
                x[i, t0 : t0 + mt, :] = 0.0

        return x


# ─────────────────────────────────────────────────────────────────────────
# Length filter (drop too-short / too-long cuts before bucketing)
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class LengthFilter:
    """Static utterance-duration filter applied to a CutSet."""

    min_seconds: float = 1.0
    max_seconds: float = 20.0

    def __call__(self, cuts):
        """Filter a Lhotse CutSet by duration. Returns the filtered set."""
        try:
            from lhotse import CutSet  # noqa: F401
        except ImportError:  # pragma: no cover
            return cuts
        return cuts.filter(lambda c: self.min_seconds <= c.duration <= self.max_seconds)


# ─────────────────────────────────────────────────────────────────────────
# Per-utterance feature mean-variance normalization
# ─────────────────────────────────────────────────────────────────────────


class PerUtteranceMVN(nn.Module):
    """Normalize each (T, F) feature matrix to zero-mean / unit-variance.

    Used by default — global CMVN is opt-in via the training config.
    """

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"PerUtteranceMVN expects (B,T,F); got {tuple(x.shape)}.")
        if lengths is None:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True, unbiased=False).clamp(min=1e-8)
            return (x - mean) / var.sqrt()

        # Length-aware version: compute stats over valid frames only.
        b, t, f = x.shape
        idx = torch.arange(t, device=x.device).unsqueeze(0)
        mask = (idx < lengths.unsqueeze(1)).unsqueeze(-1).to(dtype=x.dtype)  # (B,T,1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)                 # (B,1,1)
        mean = (x * mask).sum(dim=1, keepdim=True) / denom
        var = ((x - mean) ** 2 * mask).sum(dim=1, keepdim=True) / denom
        var = var.clamp(min=1e-8)
        out = (x - mean) / var.sqrt()
        # zero out padding to keep the masking semantic consistent downstream
        return out * mask
