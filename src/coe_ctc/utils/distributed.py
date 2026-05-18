"""Thin DDP wrappers used by the training and evaluation scripts.

Mirrors the helpers in
``tsm-trainer008_aed/.../scripts/forecasting/training/training_utils.py`` —
``get_world_size``, ``setup_distributed_seeds``, ``log_distributed_info`` etc.

Single-GPU code paths must work without a torch.distributed init.
"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Iterator


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def is_distributed() -> bool:
    """True only when torch.distributed is initialized and world_size > 1."""
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        return False
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_world_size() -> int:
    """World size — honors `WORLD_SIZE` env-var before init, then torch.distributed."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except ImportError:
        pass
    return _env_int("WORLD_SIZE", 1)


def get_rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except ImportError:
        pass
    return _env_int("RANK", 0)


def get_local_rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            # No portable per-process API; rely on env-var set by torchrun.
            pass
    except ImportError:
        pass
    return _env_int("LOCAL_RANK", 0)


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(backend: str = "nccl") -> tuple[int, int, int]:
    """Initialize torch.distributed if `WORLD_SIZE > 1`.

    Returns: ``(rank, local_rank, world_size)``.

    Safe to call once at the top of the training entry. No-op on single GPU.
    """
    import torch

    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)

    if world_size > 1:
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def teardown_distributed() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except ImportError:
        pass


def setup_seeds(seed: int) -> None:
    """Seed Python / numpy / torch (CPU + CUDA) deterministically per-rank.

    We offset the seed by ``rank`` so different GPUs get different shuffles.
    """
    rank = get_rank()
    eff = seed + rank
    random.seed(eff)
    try:
        import numpy as np

        np.random.seed(eff)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(eff)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eff)
    except ImportError:
        pass


def barrier() -> None:
    """No-op when not distributed."""
    if is_distributed():
        import torch.distributed as dist

        dist.barrier()


def unwrap_model(model):
    """Strip DDP / ``torch.compile`` wrappers and return the underlying module."""
    import torch.nn as nn

    inner = model
    while True:
        if hasattr(inner, "module") and isinstance(inner.module, nn.Module):
            inner = inner.module
            continue
        if hasattr(inner, "_orig_mod") and isinstance(inner._orig_mod, nn.Module):
            inner = inner._orig_mod
            continue
        break
    return inner


@contextmanager
def main_process_first() -> Iterator[None]:
    """Context manager: rank-0 runs first, other ranks wait at a barrier.

    Useful for downloading datasets or building manifests once.
    """
    if is_distributed() and not is_main_process():
        barrier()
    yield
    if is_distributed() and is_main_process():
        barrier()


def log_distributed_info(logger=None) -> None:
    """Print a one-line summary of the distributed setup."""
    msg = (
        f"Distributed: world_size={get_world_size()} rank={get_rank()} "
        f"local_rank={get_local_rank()} initialized={is_distributed()}"
    )
    if logger is not None:
        logger.info(msg)
    else:
        print(msg, flush=True)
