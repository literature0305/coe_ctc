"""Validation visualization — PNG plots for WER, CER, loss, and LR.

Produces these files under ``<output_dir>/plots/``:
  * ``valid_wer.png``   — WER vs step, with best-so-far line
  * ``valid_cer.png``   — CER vs step, with best-so-far line
  * ``valid_loss.png``  — validation loss vs step
  * ``train_loss.png``  — training loss vs step (raw + precomputed EMAs)
  * ``lr.png``          — learning-rate schedule

Plots are regenerated on every call; cheap enough that we can afford to do
this every validation pass without measurable overhead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

# Cap plotted points per series. Matplotlib chokes on hundreds-of-thousands
# of points (multi-MB PNG, multi-second render); subsampling keeps the curve
# shape intact and the file small.
_MAX_PLOT_POINTS = 6000


def _running_min(xs: list[float]) -> list[float]:
    """Element-wise best-so-far (running min)."""
    out: list[float] = []
    best = float("inf")
    for x in xs:
        best = min(best, x)
        out.append(best)
    return out


def _subsample(xs: Sequence) -> Sequence:
    n = len(xs)
    if n <= _MAX_PLOT_POINTS:
        return xs
    stride = max(1, n // _MAX_PLOT_POINTS)
    return xs[::stride]


def render_validation_plots(
    *,
    history: list[dict],
    lr_history: list[tuple[int, float]],
    output_dir: str | Path,
    loss_history: list[tuple[int, float]] | None = None,
    loss_ema_history: list[tuple[int, float, float]] | None = None,
) -> None:
    """Write the 4 PNGs from the accumulated validation history.

    Imports matplotlib lazily so the rest of the codebase stays importable on
    headless boxes without matplotlib installed.
    """
    output_dir = Path(output_dir) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot rendering.")
        return

    if history:
        steps = [r["step"] for r in history]
        # For CoE every history row carries per_pass; non-CoE rows have None.
        num_passes = 1
        first_with_passes = next((r for r in history if r.get("per_pass")), None)
        if first_with_passes is not None:
            num_passes = len(first_with_passes["per_pass"])

        def _per_m(metric: str) -> list[list[float]]:
            """Per-pass series; shape [m][step] in metric units (% for WER/CER)."""
            scale = 100.0 if metric in ("wer", "cer") else 1.0
            series: list[list[float]] = [[] for _ in range(num_passes)]
            for r in history:
                passes = r.get("per_pass")
                if passes:
                    for m, p in enumerate(passes):
                        series[m].append(p[metric] * scale)
                else:
                    series[0].append(r[metric] * scale)
            return series

        def _plot(metric: str, ylabel: str, title: str, fname: str, color0: str) -> None:
            series = _per_m(metric)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for m, ys in enumerate(series):
                ax.plot(steps, ys, marker="o", label=f"m={m}" if num_passes > 1 else f"dev-other {metric.upper()}")
            # best-so-far line — track the *last* pass (best-checkpoint target).
            target = series[-1]
            if metric in ("wer", "cer"):
                ax.plot(steps, _running_min(target), linestyle="--", color="C7", label="best-so-far (m=M)")
            ax.set_xlabel("step")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(str(output_dir / fname), dpi=120)
            plt.close(fig)

        _plot("wer", "WER (%)", "Validation WER", "valid_wer.png", "C0")
        _plot("cer", "CER (%)", "Validation CER", "valid_cer.png", "C2")
        _plot("loss", "loss", "Validation loss", "valid_loss.png", "C4")

    if lr_history:
        sampled = _subsample(lr_history)
        steps = [s for s, _ in sampled]
        lrs = [v for _, v in sampled]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(steps, lrs, color="C5")
        ax.set_xlabel("step")
        ax.set_ylabel("lr")
        ax.set_title("Learning rate schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(output_dir / "lr.png"), dpi=120)
        plt.close(fig)

    if loss_history:
        sampled_loss = _subsample(loss_history)
        steps = [s for s, _ in sampled_loss]
        losses = [v for _, v in sampled_loss]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(steps, losses, color="C0", alpha=0.25, linewidth=0.8, label="raw")
        if loss_ema_history:
            sampled_ema = _subsample(loss_ema_history)
            ema_steps = [s for s, _, _ in sampled_ema]
            ema_fast = [f for _, f, _ in sampled_ema]
            ema_slow = [s for _, _, s in sampled_ema]
            ax.plot(ema_steps, ema_fast, color="C0", linewidth=1.4, label="EMA α=0.1")
            ax.plot(ema_steps, ema_slow, color="C3", linewidth=1.4, label="EMA α=0.01")
        ax.set_xlabel("step")
        ax.set_ylabel("CTC loss")
        ax.set_title("Training loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(output_dir / "train_loss.png"), dpi=120)
        plt.close(fig)
