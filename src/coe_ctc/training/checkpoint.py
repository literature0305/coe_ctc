"""Checkpoint save / load / resume + top-K best tracker.

Layout under ``output_dir``::

    checkpoints/
      latest.pt                     # full training state (model+opt+sched+step+RNG)
      checkpoint-<step>.pt          # rolling N latest

    best_checkpoints/
      step={s:07d}-wer={w:.4f}.pt   # top-K best by validation WER
      manifest.json                 # bookkeeping for top-K
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from coe_ctc.utils.distributed import unwrap_model

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# State container
# ─────────────────────────────────────────────────────────────────────────


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError:
            pass
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ─────────────────────────────────────────────────────────────────────────
# Save / load
# ─────────────────────────────────────────────────────────────────────────


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    step: int = 0,
    epoch: int = 0,
    best_wer: float = float("inf"),
    config: Optional[dict] = None,
    include_rng: bool = True,
    extra: Optional[dict] = None,
) -> Path:
    """Atomic save: write to ``path.tmp`` then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model": unwrap_model(model).state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "best_wer": float(best_wer),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if include_rng:
        state["rng"] = _rng_state()
    if config is not None:
        state["config"] = config
    if extra is not None:
        state["extra"] = extra

    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(state, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: Any = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
) -> dict:
    """Load a checkpoint dict. Optionally restore model/optimizer/etc. in place.

    Returns the **full** loaded dict so the caller can read ``step``,
    ``epoch``, ``best_wer``, ``config``, ``extra`` directly.
    """
    state = torch.load(str(path), map_location=map_location, weights_only=False)

    if model is not None and "model" in state:
        msg = unwrap_model(model).load_state_dict(state["model"], strict=strict)
        logger.info(f"  Loaded model state from {path}. missing/unexpected: {msg}")
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    if restore_rng and "rng" in state:
        _set_rng_state(state["rng"])

    return state


# ─────────────────────────────────────────────────────────────────────────
# Top-K best tracker
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class BestCheckpoint:
    step: int
    wer: float
    path: str

    def to_json(self) -> dict:
        return asdict(self)


class TopKCheckpointManager:
    """Tracks the K best checkpoints by validation WER (ascending).

    Files are stored under ``output_dir/best_checkpoints/``. Worse entries are
    auto-deleted when capacity is exceeded. The manifest is a simple JSON file
    so external tools (e.g. ``checkpoint_average.py`` in Phase 4) can read it
    without instantiating this class.
    """

    def __init__(self, root_dir: str | Path, top_k: int = 5) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.top_k = int(top_k)
        self.manifest_path = self.root / "manifest.json"
        self.entries: list[BestCheckpoint] = []
        if self.manifest_path.exists():
            try:
                raw = json.loads(self.manifest_path.read_text())
                self.entries = [BestCheckpoint(**d) for d in raw]
            except Exception as exc:
                logger.warning(f"  Could not parse {self.manifest_path}: {exc}. Starting fresh.")

    def _save_manifest(self) -> None:
        self.entries.sort(key=lambda e: e.wer)
        self.manifest_path.write_text(json.dumps([e.to_json() for e in self.entries], indent=2))

    def consider(
        self,
        *,
        step: int,
        wer: float,
        model: torch.nn.Module,
        config: Optional[dict] = None,
    ) -> Optional[Path]:
        """If ``wer`` improves on the worst kept entry, save the model and prune.

        Returns the path of the newly-written checkpoint (or ``None``).
        """
        if not _isfinite(wer):
            return None
        if len(self.entries) >= self.top_k and wer >= self.entries[-1].wer:
            return None

        fname = f"step={step:07d}-wer={wer:.4f}.pt"
        path = self.root / fname
        # Save model-only (no optimizer) for compact best checkpoints.
        torch.save({"model": unwrap_model(model).state_dict(), "step": int(step), "wer": float(wer), "config": config}, path)
        self.entries.append(BestCheckpoint(step=int(step), wer=float(wer), path=str(path)))

        # Prune
        self.entries.sort(key=lambda e: e.wer)
        while len(self.entries) > self.top_k:
            worst = self.entries.pop()
            try:
                Path(worst.path).unlink(missing_ok=True)
            except OSError:
                pass
        self._save_manifest()
        return path

    @property
    def best(self) -> Optional[BestCheckpoint]:
        return self.entries[0] if self.entries else None


def _isfinite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))
