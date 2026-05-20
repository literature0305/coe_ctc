"""Curriculum learning + OOM pre-flight for CTC training.

CTC has a long-standing instability: in the first few thousand steps the
blank-collapse minimum is very close at hand, and one bad batch (long
utterance with under-determined alignment) can push the loss into the
all-blank attractor. Curriculum learning side-steps this by training on
*short* utterances first, where the alignment-to-frames ratio is high and
gradient direction is unambiguous, then progressively admitting longer
utterances as the model develops a stable alignment prior.

Schedule shape (from the training YAML, in the ``curriculum:`` section):

  curriculum:
    cl_schedule_len:  [2, 4, 8]            # max-duration cap (sec) per phase
    cl_schedule_step: [1000, 2000, 4000]   # step boundaries (exclusive upper)

means:

  step  0..   999 : sample durations in [min_seconds, 2]s
  step 1000..1999 : sample durations in [min_seconds, 4]s
  step 2000..3999 : sample durations in [min_seconds, 8]s
  step 4000..     : sample durations in [min_seconds, data.max_seconds]

The trainer rebuilds the train DataLoader on every phase boundary; the
sampler/dataset cost is small relative to one curriculum phase.

OOM pre-flight
--------------
Because curriculum starts with very short utterances, the first
*real-length* batch is delayed until the last phase boundary. If the
configured ``data.max_duration`` / ``data.max_seconds`` won't fit in GPU
memory, we want to know in seconds — not after 4k wasted steps. The
pre-flight runs ONE worst-case forward+backward (batch shaped to the full
``max_seconds`` × ``max_duration / max_seconds``) on each rank before the
training loop starts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from coe_ctc.data.fbank import FbankConfig
from coe_ctc.models.subsampling import Conv2dSubsampling

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Curriculum schedule
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class CurriculumConfig:
    cl_schedule_len: List[float] = field(default_factory=list)
    cl_schedule_step: List[int] = field(default_factory=list)

    def is_active(self) -> bool:
        return len(self.cl_schedule_len) > 0

    def validate(self, *, min_seconds: float, max_seconds: float) -> None:
        if len(self.cl_schedule_len) != len(self.cl_schedule_step):
            raise ValueError(
                "curriculum.cl_schedule_len and cl_schedule_step must have the same length; "
                f"got {len(self.cl_schedule_len)} vs {len(self.cl_schedule_step)}."
            )
        if not self.is_active():
            return
        prev_step = 0
        for s in self.cl_schedule_step:
            if s <= prev_step:
                raise ValueError(
                    "curriculum.cl_schedule_step must be strictly increasing positive ints; "
                    f"got {self.cl_schedule_step}."
                )
            prev_step = s
        for prev_l, cur_l in zip(self.cl_schedule_len[:-1], self.cl_schedule_len[1:]):
            if cur_l < prev_l:
                raise ValueError(
                    "curriculum.cl_schedule_len should be non-decreasing; "
                    f"got {self.cl_schedule_len}."
                )
        if self.cl_schedule_len[0] < min_seconds:
            raise ValueError(
                f"curriculum.cl_schedule_len[0]={self.cl_schedule_len[0]} is below "
                f"data.min_seconds={min_seconds}; phase 0 would have no cuts."
            )
        if self.cl_schedule_len[-1] > max_seconds:
            logger.warning(
                "curriculum.cl_schedule_len[-1]=%.2f exceeds data.max_seconds=%.2f; "
                "the last curriculum phase will effectively use data.max_seconds.",
                self.cl_schedule_len[-1], max_seconds,
            )

    def phase_index(self, step: int) -> int:
        for i, boundary in enumerate(self.cl_schedule_step):
            if step < boundary:
                return i
        return len(self.cl_schedule_step)

    def max_seconds_for(self, step: int, *, fallback: float) -> float:
        idx = self.phase_index(step)
        if idx >= len(self.cl_schedule_len):
            return float(fallback)
        return float(self.cl_schedule_len[idx])

    def describe(self, *, fallback: float) -> str:
        if not self.is_active():
            return "off"
        parts = []
        lo = 0
        for length, hi in zip(self.cl_schedule_len, self.cl_schedule_step):
            parts.append(f"[{lo}..{hi}):≤{length:g}s")
            lo = hi
        parts.append(f"[{lo}..):≤{fallback:g}s")
        return " ".join(parts)


def parse_curriculum_config(cfg_dict: Optional[dict]) -> CurriculumConfig:
    src = cfg_dict or {}
    return CurriculumConfig(
        **{k: v for k, v in src.items() if k in CurriculumConfig.__dataclass_fields__}
    )


# ─────────────────────────────────────────────────────────────────────────
# OOM pre-flight
# ─────────────────────────────────────────────────────────────────────────


def oom_preflight(
    *,
    model: nn.Module,
    device: torch.device,
    num_features: int,
    max_seconds: float,
    max_duration: float,
    vocab_size: int,
    blank_idx: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_accum_steps: int = 1,
    mvn: Optional[nn.Module] = None,
    spec_aug: Optional[nn.Module] = None,
) -> dict:
    """Run one worst-case forward+backward to surface OOMs early.

    The synthetic batch is shaped so each utterance is exactly ``max_seconds``
    long and the batch size is ``floor(max_duration / max_seconds)`` — the
    largest batch the bucketing sampler can yield. Targets are random
    non-blank tokens with ``U = encoded_len // 2`` (safely below the
    post-subsampling length the encoder produces).

    Pass the *unwrapped* model — going through DDP triggers a cross-rank
    all-reduce that is unrelated to per-GPU memory usage.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        logger.info("[oom-preflight] non-CUDA device; skipping.")
        return {"skipped": True}

    frames_per_sec = 1.0 / FbankConfig().frame_shift
    T = max(8, int(round(max_seconds * frames_per_sec)))
    B = max(1, int(max_duration // max(1e-6, max_seconds)))
    encoded_len = int(Conv2dSubsampling.out_length(T))
    U = max(1, encoded_len // 2)

    logger.info(
        "[oom-preflight] worst-case probe: B=%d, T=%d frames (~%.1fs), U=%d tokens "
        "(max_duration=%.1fs)",
        B, T, max_seconds, U, max_duration,
    )

    features = torch.randn(B, T, num_features, device=device, dtype=torch.float32)
    feature_lengths = torch.full((B,), T, dtype=torch.long, device=device)
    lo = max(1, blank_idx + 1)
    hi = max(lo + 1, vocab_size)
    tokens = torch.randint(low=lo, high=hi, size=(B, U), dtype=torch.long, device=device)
    target_lengths = torch.full((B,), U, dtype=torch.long, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    was_training = model.training
    model.train()
    out = None
    loss = None
    try:
        if mvn is not None:
            features = mvn(features, feature_lengths)
        if spec_aug is not None:
            features = spec_aug(features, feature_lengths)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(features, feature_lengths, targets=tokens, target_lengths=target_lengths)
            loss = out.loss / max(1, grad_accum_steps)
        loss.backward()
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    except torch.cuda.OutOfMemoryError:
        logger.error(
            "[oom-preflight] FAILED with OOM at probe shape B=%d T=%d U=%d. "
            "Try one of:\n"
            "  • lower data.max_duration\n"
            "  • lower data.max_seconds\n"
            "  • enable AMP (amp.enabled: true, dtype: bfloat16)\n"
            "  • raise optim.grad_accum_steps (keeps effective batch, lowers peak)\n"
            "  • for CoE: set coe.pre_enc_feature_detach: true",
            B, T, U,
        )
        raise
    finally:
        # Release autograd graph + activations before the training loop's first
        # forward — caller's optimizer.zero_grad(set_to_none=True) handles grads.
        del out, loss, features, feature_lengths, tokens, target_lengths
        if not was_training:
            model.eval()
        torch.cuda.empty_cache()

    logger.info(
        "[oom-preflight] OK. peak_allocated=%.2f GiB, peak_reserved=%.2f GiB.",
        peak_alloc, peak_reserved,
    )
    return {
        "B": B,
        "T": T,
        "U": U,
        "peak_alloc_gib": peak_alloc,
        "peak_reserved_gib": peak_reserved,
    }
