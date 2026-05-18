"""CNN subsampling front-end: ¼ time reduction.

Matches IceFall's ``Conv2dSubsampling`` semantics:
    input  (B, T,  num_mel_bins)
    output (B, T', d_model),  T' = ((T - 1) // 2 - 1) // 2

That is, two stride-2 convolutions reduce the time axis by 4×. The frequency
axis is collapsed by a linear projection.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dSubsampling(nn.Module):
    """Standard 2-layer Conv2d subsampling.

    Args:
        in_channels: Input feature dim (e.g. ``80`` for fBank-80).
        out_channels: Encoder model dim ``d_model``.
        dropout: Dropout applied after the linear projection.
        layer1_channels: Output channels of the first conv (default 32 ×).
        layer2_channels: Output channels of the second conv (default ``out_channels``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dropout: float = 0.1,
        layer1_channels: int = 32,
        layer2_channels: int | None = None,
    ) -> None:
        super().__init__()
        layer2_channels = layer2_channels or out_channels

        # Both convs use kernel=3, stride=2 → each reduces (T, F) by ~2×.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=layer1_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=layer1_channels, out_channels=layer2_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        # After two stride-2 convs, the frequency axis is:
        #   F0 = in_channels
        #   F1 = (F0 - 1) // 2     # post conv1
        #   F2 = (F1 - 1) // 2     # post conv2
        # We collapse (F2 × layer2_channels) → out_channels via a single linear.
        f1 = (in_channels - 1) // 2
        f2 = (f1 - 1) // 2
        if f2 <= 0:
            raise ValueError(
                f"in_channels={in_channels} is too small for two stride-2 convs (residual freq={f2})."
            )
        self._post_conv_freq = f2
        self.out_dim = out_channels

        self.out = nn.Linear(layer2_channels * f2, out_channels)
        self.out_norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def out_length(in_length: torch.Tensor | int) -> torch.Tensor | int:
        """Compute the post-subsampling time length, matching the conv arithmetic above."""
        if isinstance(in_length, torch.Tensor):
            t1 = (in_length - 1).div(2, rounding_mode="floor")
            t2 = (t1 - 1).div(2, rounding_mode="floor")
            return t2.clamp(min=0)
        t1 = (in_length - 1) // 2
        t2 = (t1 - 1) // 2
        return max(0, t2)

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:     (B, T, in_channels)
            x_lens: (B,) int64, time lengths before subsampling.

        Returns:
            (out, out_lens) where ``out`` has shape (B, T', out_channels).
        """
        # (B, T, F) → (B, 1, T, F) for Conv2d
        x = x.unsqueeze(1)
        x = self.conv(x)
        # x: (B, C_out, T', F')
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        x = self.out(x)
        x = self.out_norm(x)
        x = self.dropout(x)
        out_lens = self.out_length(x_lens)
        return x, out_lens
