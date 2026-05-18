"""Average the top-K best checkpoints by validation WER.

Reads ``best_checkpoints/manifest.json`` (written by
:class:`coe_ctc.training.checkpoint.TopKCheckpointManager`) and writes
``model.avg{N}.steps=<sorted-steps>.pt`` into the same directory.

CLI:
    python -m coe_ctc.decoding.checkpoint_average \
        --models outputs/ctc_conformer_small_bpe3000/best_checkpoints \
        --n 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import torch

logger = logging.getLogger(__name__)


def _read_manifest(best_dir: Path) -> list[dict]:
    manifest = best_dir / "manifest.json"
    if not manifest.is_file():
        # Fallback: scan filenames "step=*-wer=*.pt".
        entries = []
        for p in sorted(best_dir.glob("step=*-wer=*.pt")):
            try:
                stem = p.stem  # "step=0012345-wer=0.0345"
                step = int(stem.split("step=")[1].split("-")[0])
                wer = float(stem.split("wer=")[1])
                entries.append({"step": step, "wer": wer, "path": str(p)})
            except Exception:
                continue
        return sorted(entries, key=lambda e: e["wer"])
    raw = json.loads(manifest.read_text())
    return sorted(raw, key=lambda e: e["wer"])


def select_top_n(best_dir: str | os.PathLike, n: int) -> list[Path]:
    """Return up to *n* checkpoint paths, ascending by WER."""
    entries = _read_manifest(Path(best_dir))
    return [Path(e["path"]) for e in entries[: int(n)]]


def average_state_dicts(paths: Iterable[Path]) -> dict[str, torch.Tensor]:
    """Uniform-mean state dicts (assumes identical keys)."""
    paths = list(paths)
    if not paths:
        raise ValueError("No checkpoints to average.")
    accum: dict[str, torch.Tensor] = {}
    # Remember the original dtype of each tensor as we see it the first time,
    # so we can cast the mean back without reloading any checkpoint.
    dtypes: dict[str, torch.dtype] = {}
    for p in paths:
        state = torch.load(str(p), map_location="cpu", weights_only=False)
        sd = state.get("model", state)
        for k, v in sd.items():
            if not torch.is_tensor(v):
                continue
            if k not in accum:
                accum[k] = v.detach().clone().to(torch.float64)
                dtypes[k] = v.dtype
            else:
                if accum[k].shape != v.shape:
                    raise RuntimeError(f"Shape mismatch for {k} in {p}: {accum[k].shape} vs {v.shape}")
                accum[k].add_(v.detach().to(torch.float64))
    n = len(paths)
    return {k: (v / n).to(dtypes.get(k, torch.float32)) for k, v in accum.items()}


def ensure_averaged_checkpoint(best_dir: str | os.PathLike, n: int) -> Path:
    """If the averaged file already exists, reuse it; otherwise compute + save.

    The filename encodes which source steps were averaged so cache hits are
    obvious when comparing runs.
    """
    best_dir = Path(best_dir)
    paths = select_top_n(best_dir, n)
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {best_dir}.")

    # Build filename from the included steps.
    steps = []
    for p in paths:
        try:
            steps.append(int(p.stem.split("step=")[1].split("-")[0]))
        except Exception:
            pass
    steps.sort()
    stem = "-".join(f"{s:07d}" for s in steps)
    out_path = best_dir / f"model.avg{len(paths)}.steps={stem}.pt"

    if out_path.is_file():
        logger.info(f"  [reuse] averaged checkpoint exists: {out_path}")
        return out_path

    logger.info(f"Averaging {len(paths)} checkpoints → {out_path.name}")
    sd = average_state_dicts(paths)
    tmp = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    torch.save({"model": sd, "averaged_from": [str(p) for p in paths]}, tmp)
    tmp.replace(out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────


def _main() -> int:
    p = argparse.ArgumentParser(description="Average top-N best CTC checkpoints by WER.")
    p.add_argument("--models", required=True, help="best_checkpoints/ directory.")
    p.add_argument("--n", type=int, default=5, help="Number of checkpoints to average (default 5).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    out = ensure_averaged_checkpoint(args.models, args.n)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
