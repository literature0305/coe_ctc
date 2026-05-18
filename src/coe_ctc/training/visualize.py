"""Validation visualization — PNG plots for WER, CER, loss, and LR.

Produces these files under ``<output_dir>/plots/``:
  * ``valid_wer.png``   — WER vs step, with best-so-far line
  * ``valid_cer.png``   — CER vs step, with best-so-far line
  * ``valid_loss.png``  — validation loss vs step
  * ``lr.png``          — learning-rate schedule

Plots are regenerated on every call; cheap enough that we can afford to do
this every validation pass without measurable overhead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def _running_min(xs: list[float]) -> list[float]:
    """Element-wise best-so-far (running min)."""
    out: list[float] = []
    best = float("inf")
    for x in xs:
        best = min(best, x)
        out.append(best)
    return out


def render_validation_plots(
    *,
    history: list[dict],
    lr_history: list[tuple[int, float]],
    output_dir: str | Path,
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
        steps = [s for s, _ in lr_history]
        lrs = [v for _, v in lr_history]
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
