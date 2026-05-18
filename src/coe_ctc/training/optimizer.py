"""Optimizer + LR scheduler factories.

Default optimizer is ``AdamW`` (fused on CUDA when available). The scheduler
is a Noam-style warmup-then-decay or a warmup+cosine — both are standard
for ASR. ``ScaledAdam`` / ``Eden`` (zipformer paper) are intentionally NOT
implemented here; we keep things framework-agnostic. Users who want full
zipformer parity can swap them in via ``--optim-class``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass
class OptimConfig:
    """Knobs read from YAML's ``optim:`` block."""

    name: str = "adamw"           # "adamw" | "adam"
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-9
    weight_decay: float = 1e-2
    fused: bool = True            # uses CUDA fused kernel if available

    # Scheduler
    scheduler: str = "warmup_cosine"  # "warmup_cosine" | "noam" | "warmup_inv_sqrt" | "constant"
    warmup_steps: int = 10000
    max_steps: int = 200000
    min_lr_ratio: float = 0.01    # cosine floor as fraction of peak LR
    noam_d_model: int = 512       # only for "noam"


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    """AdamW with weight-decay only on tensors with ``ndim >= 2`` (no decay on biases/norms)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "bias" not in n.lower():
            decay.append(p)
        else:
            no_decay.append(p)
    param_groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    fused_supported = cfg.fused and torch.cuda.is_available()
    name = cfg.name.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            fused=fused_supported,
        )
    if name == "adam":
        return torch.optim.Adam(
            param_groups,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            fused=fused_supported,
        )
    raise ValueError(f"Unknown optimizer '{cfg.name}'.")


# ─────────────────────────────────────────────────────────────────────────
# Schedulers
# ─────────────────────────────────────────────────────────────────────────


def _warmup_cosine_lambda(step: int, warmup: int, total: int, min_ratio: float) -> float:
    if step < warmup:
        return float(step) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cos_decay


def _warmup_inv_sqrt_lambda(step: int, warmup: int) -> float:
    if step < warmup:
        return float(step) / max(1, warmup)
    return math.sqrt(warmup / max(1, step))


def _noam_lambda(step: int, warmup: int, d_model: int) -> float:
    step = max(1, step)
    # Noam factor; multiply by d_model^-0.5 here so the peak LR is consistent
    # with what users set in YAML (i.e. lr stays roughly cfg.lr in size).
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: OptimConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    name = cfg.scheduler.lower()
    if name == "warmup_cosine":
        fn = lambda s: _warmup_cosine_lambda(s, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio)
    elif name == "warmup_inv_sqrt":
        fn = lambda s: _warmup_inv_sqrt_lambda(s, cfg.warmup_steps)
    elif name == "noam":
        fn = lambda s: _noam_lambda(s, cfg.warmup_steps, cfg.noam_d_model)
    elif name == "constant":
        fn = lambda s: 1.0
    else:
        raise ValueError(f"Unknown scheduler '{cfg.scheduler}'.")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)
