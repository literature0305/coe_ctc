"""Training Health Report — visually identical to errlog019-2_output_example.

Drop-in callback used by :mod:`coe_ctc.training.train`. Tracks loss/grad/lr
/throughput/memory/data statistics and prints a box-drawn report every N
training steps. The same instance also stores the last validation
``BENCHMARK EVALUATION`` payload for inclusion in the next health report.
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import torch

from coe_ctc.training.checkpoint import _isfinite
from coe_ctc.utils.logging import (
    box_bottom,
    box_inner_bottom,
    box_inner_top,
    box_line,
    box_section_title,
    box_top,
    format_duration,
    make_progress_bar,
    make_sparkline,
)


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────


def _ema(prev: float, current: float, alpha: float) -> float:
    if prev is None or not _isfinite(prev):
        return float(current)
    return alpha * float(current) + (1.0 - alpha) * float(prev)


# ─────────────────────────────────────────────────────────────────────────
# Snapshot dataclass — Bytes returned by validation that get folded into
# the next health report's BENCHMARK section.
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class BenchmarkSnapshot:
    step: int
    wer: float
    cer: float
    loss: float
    best_wer: float
    best_wer_step: int
    elapsed_sec: float
    samples: int = 0


# ─────────────────────────────────────────────────────────────────────────
# Main reporter
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class HealthReporterConfig:
    interval_steps: int = 50
    log_dir: Optional[str] = None
    grad_window: int = 200          # rolling window for grad-norm stats
    sparkline_window: int = 40


class TrainingHealthReporter:
    """Aggregates training telemetry and renders box-format reports.

    Usage from the training loop::

        reporter = TrainingHealthReporter(cfg, max_steps=200_000)
        reporter.log_config(model, dm_cfg, optim_cfg, world_size=4)
        for step, batch in enumerate(loader, start=1):
            loss = ...; grad_norm = ...; lr = ...
            reporter.step_update(loss=loss, grad_norm=grad_norm, clipped=clipped,
                                 lr=lr, batch_frames=..., batch_tokens=..., samples=batch_size)
            if step % cfg.interval_steps == 0:
                reporter.emit_report(step, max_steps)

    Validation callbacks should call ``record_benchmark()`` to leave a
    snapshot for the next report.
    """

    def __init__(self, cfg: HealthReporterConfig, max_steps: int) -> None:
        self.cfg = cfg
        self.max_steps = max(1, int(max_steps))

        # Loss state — un-bounded so the LR / train-loss plots span the whole run.
        self.loss_history: list[tuple[int, float]] = []
        self.loss_ema_history: list[tuple[int, float, float]] = []  # (step, fast, slow)
        self.loss_ema_fast: float = float("nan")
        self.loss_ema_slow: float = float("nan")
        self.best_loss: float = float("inf")
        self.best_loss_step: int = 0
        self.loss_milestones: list[tuple[int, float]] = []
        self._last_milestone_loss: float = float("inf")

        # Grad state
        self.grad_norms: deque[float] = deque(maxlen=cfg.grad_window)
        self.clip_count: int = 0
        self.grad_count: int = 0
        self.nan_count: int = 0
        self.explosion_count: int = 0

        self.lr_history: list[tuple[int, float]] = []
        self.peak_lr: float = 0.0

        # Throughput
        self._wall_start: float = time.time()
        self._step_times: deque[float] = deque(maxlen=200)
        self._last_step_t: Optional[float] = None
        self.total_samples: int = 0
        self.total_frames: int = 0
        self.total_tokens: int = 0
        self._recent_step_window = 50

        # Time breakdown (seconds)
        self.train_seconds: float = 0.0
        self.validation_seconds: float = 0.0
        self.visualization_seconds: float = 0.0
        self.save_seconds: float = 0.0
        self.curriculum_seconds: float = 0.0
        self.curriculum_rebuilds: int = 0

        # Validation stash
        self.last_benchmark: Optional[BenchmarkSnapshot] = None

        # Dataset stats since last report
        self._batches_this_window: int = 0
        self._frames_this_window: int = 0
        self._tokens_this_window: int = 0
        self._input_len_avg: float = 0.0
        self._input_len_max: int = 0
        self._target_len_avg: float = 0.0
        self._target_len_max: int = 0
        self._pad_ratio_sum: float = 0.0
        self._loss_milestone_interval: int = 100

        # Effective-batch tracking (utterances per optimizer step). Populated
        # by train.py via ``step_update(effective_batch_utterances=...)`` and
        # the public ``grad_accum_steps`` / ``coe_passes`` fields below.
        self._eff_batch_window: deque[int] = deque(maxlen=200)
        self.grad_accum_steps: int = 1
        self.coe_passes: int = 1

    # =====================================================================
    # Public API: configuration banner
    # =====================================================================

    def log_config(
        self,
        *,
        model_summary: dict[str, Any],
        data_summary: dict[str, Any],
        optim_summary: dict[str, Any],
        distributed_summary: dict[str, Any],
        monitoring_summary: dict[str, Any],
    ) -> None:
        """Print the one-time TRAINING CONFIGURATION banner at start of run."""
        lines: list[str] = [box_top("CTC ASR TRAINING CONFIGURATION")]

        # Model
        lines.append(box_inner_top("Model"))
        for k, v in model_summary.items():
            lines.append(box_line(f"{k:<18s}{v}"))
        lines.append(box_inner_bottom())

        # Data
        lines.append(box_inner_top("Data"))
        for k, v in data_summary.items():
            lines.append(box_line(f"{k:<18s}{v}"))
        lines.append(box_inner_bottom())

        # Optimization
        lines.append(box_inner_top("Optimization"))
        for k, v in optim_summary.items():
            lines.append(box_line(f"{k:<18s}{v}"))
        lines.append(box_inner_bottom())

        # Distributed
        lines.append(box_inner_top("Distributed"))
        for k, v in distributed_summary.items():
            lines.append(box_line(f"{k:<18s}{v}"))
        lines.append(box_inner_bottom())

        # Monitoring
        lines.append(box_inner_top("Monitoring"))
        for k, v in monitoring_summary.items():
            lines.append(box_line(f"{k:<18s}{v}"))
        lines.append(box_inner_bottom())

        lines.append(box_bottom())
        for ln in lines:
            logger.info(ln)

    # =====================================================================
    # Public API: per-step update
    # =====================================================================

    def step_update(
        self,
        *,
        step: int,
        loss: float,
        grad_norm: Optional[float],
        clipped: bool,
        lr: float,
        batch_size: int,
        batch_frames: int,
        batch_tokens: int,
        input_lens_sum: int,
        input_lens_max: int,
        target_lens_sum: int,
        target_lens_max: int,
        pad_frames_ratio: float,
        effective_batch_utterances: Optional[int] = None,
    ) -> None:
        now = time.time()
        if self._last_step_t is not None:
            self._step_times.append(now - self._last_step_t)
            self.train_seconds += now - self._last_step_t
        self._last_step_t = now

        # Loss EMAs
        if _isfinite(loss):
            self.loss_history.append((int(step), float(loss)))
            self.loss_ema_fast = _ema(self.loss_ema_fast, loss, alpha=0.1)
            self.loss_ema_slow = _ema(self.loss_ema_slow, loss, alpha=0.01)
            self.loss_ema_history.append((int(step), self.loss_ema_fast, self.loss_ema_slow))
            if loss < self.best_loss:
                self.best_loss = float(loss)
                self.best_loss_step = int(step)
            # Milestones every N steps if loss improved by >5%.
            if (
                len(self.loss_history) % self._loss_milestone_interval == 0
                and loss < self._last_milestone_loss * 0.95
            ):
                self.loss_milestones.append((int(step), float(loss)))
                self._last_milestone_loss = float(loss)

        # Gradient stats
        if grad_norm is not None and _isfinite(grad_norm):
            self.grad_norms.append(float(grad_norm))
            self.grad_count += 1
            if clipped:
                self.clip_count += 1
            if grad_norm > 1000.0:
                self.explosion_count += 1
        elif grad_norm is not None and not _isfinite(grad_norm):
            self.nan_count += 1
            self.grad_count += 1

        # LR
        self.lr_history.append((int(step), float(lr)))
        if lr > self.peak_lr:
            self.peak_lr = float(lr)

        # Throughput counters
        self.total_samples += int(batch_size)
        self.total_frames += int(batch_frames)
        self.total_tokens += int(batch_tokens)

        if effective_batch_utterances is not None:
            self._eff_batch_window.append(int(effective_batch_utterances))

        # Per-window dataset stats
        self._batches_this_window += 1
        self._frames_this_window += int(batch_frames)
        self._tokens_this_window += int(batch_tokens)
        # Running averages within the window
        b = self._batches_this_window
        self._input_len_avg = ((b - 1) * self._input_len_avg + input_lens_sum / max(1, batch_size)) / b
        self._input_len_max = max(self._input_len_max, int(input_lens_max))
        self._target_len_avg = ((b - 1) * self._target_len_avg + target_lens_sum / max(1, batch_size)) / b
        self._target_len_max = max(self._target_len_max, int(target_lens_max))
        self._pad_ratio_sum += float(pad_frames_ratio)

    def record_benchmark(self, snap: BenchmarkSnapshot) -> None:
        self.last_benchmark = snap

    def add_validation_time(self, seconds: float) -> None:
        self.validation_seconds += float(seconds)

    def add_visualization_time(self, seconds: float) -> None:
        self.visualization_seconds += float(seconds)

    def add_save_time(self, seconds: float) -> None:
        self.save_seconds += float(seconds)

    def add_curriculum_time(self, seconds: float) -> None:
        self.curriculum_seconds += float(seconds)
        self.curriculum_rebuilds += 1

    # =====================================================================
    # Report emission
    # =====================================================================

    def emit_report(self, step: int, max_steps: int | None = None) -> str:
        """Render a TRAINING HEALTH REPORT and log it. Returns the rendered text."""
        if max_steps is None:
            max_steps = self.max_steps
        elapsed = time.time() - self._wall_start
        pct = step / max(1, max_steps)
        lines: list[str] = [box_top(f"TRAINING HEALTH REPORT — Step {step}/{max_steps}")]
        lines.append(
            box_section_title(
                f"{make_progress_bar(pct, width=40)} {pct*100:.1f}%  Elapsed: {format_duration(elapsed)}"
            )
        )
        # ---------- Loss ----------
        lines.append(box_inner_top("Loss Analysis"))
        cur_loss = self.loss_history[-1][1] if self.loss_history else float("nan")
        lines.append(box_line(f"Current     : {cur_loss:.6f}"))
        lines.append(box_line(f"EMA (fast)  : {self.loss_ema_fast:.6f}"))
        lines.append(box_line(f"EMA (slow)  : {self.loss_ema_slow:.6f}"))
        lines.append(box_line(f"Best        : {self.best_loss:.6f} (step {self.best_loss_step})"))
        trend_label, trend_pct = self._loss_trend()
        lines.append(box_line(f"Trend       : {trend_label} ({trend_pct:+.2f}%)"))
        variance = self._loss_variance()
        lines.append(box_line(f"Variance    : {variance:.6f}  (stability indicator)"))
        conv_rate = self._convergence_rate()
        lines.append(box_line(f"Conv. rate  : {conv_rate:+.4f} per 1K steps"))
        if self.loss_milestones:
            ms = self.loss_milestones[-1]
            lines.append(box_line(f"Milestones  : {len(self.loss_milestones)} (last: step {ms[0]} = {ms[1]:.4f})"))
        # Spark
        recent = [l for _, l in list(self.loss_history)[-self.cfg.sparkline_window:]]
        if recent:
            lines.append(box_line(f"History     : {make_sparkline(recent)}"))
        lines.append(box_inner_bottom())

        # ---------- Gradient ----------
        lines.append(box_inner_top("Gradient Health"))
        if self.grad_norms:
            gns = list(self.grad_norms)
            g_avg = sum(gns) / len(gns)
            g_max = max(gns)
            g_min = min(gns)
            g_std = statistics.stdev(gns) if len(gns) >= 2 else 0.0
            lines.append(box_line(f"Norm (avg)  : {g_avg:.4f}   Norm (max): {g_max:.4f}   Std: {g_std:.4f}"))
            lines.append(box_line(f"Norm (min)  : {g_min:.4f}"))
            clip_pct = 100.0 * self.clip_count / max(1, self.grad_count)
            warn = "  ⚠ High clip rate; consider raising max_grad_norm" if clip_pct > 80 else ""
            lines.append(box_line(f"Clip rate   : {self.clip_count}/{self.grad_count} ({clip_pct:.1f}%){warn}"))
            if self.nan_count:
                lines.append(box_line(f"NaN/Inf     : {self.nan_count}  ⚠ check loss scaling / LR / SpecAug"))
            if self.explosion_count:
                lines.append(box_line(f"Explosions  : {self.explosion_count} (norm > 1000)"))
        else:
            lines.append(box_line("(no gradient data yet)"))
        lines.append(box_inner_bottom())

        # ---------- LR ----------
        lines.append(box_inner_top("Learning Rate"))
        cur_lr = self.lr_history[-1][1] if self.lr_history else 0.0
        ratio = (cur_lr / self.peak_lr) * 100.0 if self.peak_lr > 0 else 0.0
        lines.append(box_line(f"Current     : {cur_lr:.2e}  (peak: {self.peak_lr:.2e}, ratio: {ratio:.1f}%)"))
        lr_recent = [v for _, v in list(self.lr_history)[-self.cfg.sparkline_window:]]
        if lr_recent:
            lines.append(box_line(f"Schedule    : {make_sparkline(lr_recent)}"))
        lines.append(box_inner_bottom())

        # ---------- GPU Memory ----------
        gpu = self._gpu_memory()
        lines.append(box_inner_top("GPU Memory (Rank 0)"))
        if gpu is not None:
            bar = make_progress_bar(gpu["frac"], width=30)
            lines.append(box_line(f"{bar} {gpu['allocated_gb']:.1f}/{gpu['total_gb']:.1f} GB ({gpu['frac']*100:.0f}%)"))
            lines.append(box_line(f"Allocated   : {gpu['allocated_gb']:.2f} GB   Reserved: {gpu['reserved_gb']:.2f} GB   Peak: {gpu['peak_gb']:.2f} GB"))
            if gpu["frac"] < 0.30:
                lines.append(box_line("ⓘ Low utilization — consider increasing batch_size or max_duration"))
        else:
            lines.append(box_line("(CUDA not available)"))
        lines.append(box_inner_bottom())

        # ---------- Throughput ----------
        lines.append(box_inner_top("Throughput"))
        recent_st = list(self._step_times)
        recent_sps = (1.0 / (sum(recent_st[-50:]) / max(1, len(recent_st[-50:])))) if recent_st else 0.0
        overall_sps = step / max(1.0, elapsed)
        samples_sec = self.total_samples / max(1.0, elapsed)
        frames_sec = self.total_frames / max(1.0, elapsed)
        lines.append(box_line(f"Steps/sec   : {recent_sps:.2f} (recent)    {overall_sps:.2f} (overall)"))
        lines.append(box_line(f"Samples/sec : {samples_sec:.1f}   Frames/sec: {frames_sec:.0f}"))
        if recent_st:
            avg_st = sum(recent_st) / len(recent_st)
            quartiles = statistics.quantiles(recent_st, n=100, method="inclusive") if len(recent_st) >= 2 else [recent_st[0]] * 99
            p50 = quartiles[49]
            p95 = quartiles[94]
            lines.append(box_line(f"Step time   : avg={avg_st:.3f}s  p50={p50:.3f}s  p95={p95:.3f}s"))
        eta = self._eta(step, max_steps, elapsed)
        lines.append(box_line(f"ETA         : {format_duration(eta)}  (long-term, incl. eval/save)"))
        if self._eff_batch_window:
            eff_avg = sum(self._eff_batch_window) / len(self._eff_batch_window)
            eff_line = f"Eff. batch  : {eff_avg:.1f} utt/step"
            if self.grad_accum_steps > 1:
                eff_line += f"  ({eff_avg / self.grad_accum_steps:.1f} × accum {self.grad_accum_steps})"
            if self.coe_passes > 1:
                eff_line += f"  ×{self.coe_passes} CoE passes"
            lines.append(box_line(eff_line))
        lines.append(box_line(f"Total proc  : samples={self.total_samples:,}  frames={self.total_frames:,}  tokens={self.total_tokens:,}"))
        lines.append(box_inner_bottom())

        # ---------- Time Breakdown ----------
        lines.append(box_inner_top("Time Breakdown"))
        total_known = (
            self.train_seconds + self.validation_seconds + self.visualization_seconds
            + self.save_seconds + self.curriculum_seconds
        )
        total = max(total_known, elapsed)
        def _pct(x): return 100.0 * x / max(1.0, total)
        lines.append(box_line(f"Training    : {format_duration(self.train_seconds):>10s}  ({_pct(self.train_seconds):5.1f}%)"))
        lines.append(box_line(f"Validation  : {format_duration(self.validation_seconds):>10s}  ({_pct(self.validation_seconds):5.1f}%)"))
        lines.append(box_line(f"Visualize   : {format_duration(self.visualization_seconds):>10s}  ({_pct(self.visualization_seconds):5.1f}%)"))
        lines.append(box_line(f"Save        : {format_duration(self.save_seconds):>10s}  ({_pct(self.save_seconds):5.1f}%)"))
        curr_label = (
            f"Curriculum  : {format_duration(self.curriculum_seconds):>10s}  "
            f"({_pct(self.curriculum_seconds):5.1f}%)  rebuilds={self.curriculum_rebuilds}"
        )
        lines.append(box_line(curr_label))
        lines.append(box_line(f"Total       : {format_duration(elapsed):>10s}"))
        lines.append(box_inner_bottom())

        # ---------- Data Pipeline ----------
        lines.append(box_inner_top("Data Pipeline"))
        lines.append(box_line(f"Workers     : {os.environ.get('OMP_NUM_THREADS', '?')} OMP / {os.environ.get('MKL_NUM_THREADS', '?')} MKL"))
        lines.append(box_line(f"Total frames: {self.total_frames:,}"))
        lines.append(box_line(f"Total tokens: {self.total_tokens:,}"))
        lines.append(box_inner_bottom())

        # ---------- Last Benchmark ----------
        if self.last_benchmark is not None:
            snap = self.last_benchmark
            lines.append(box_inner_top("Last Validation (dev-other)"))
            lines.append(box_line(f"Step        : {snap.step}"))
            lines.append(box_line(f"WER         : {snap.wer*100:.2f}%   CER: {snap.cer*100:.2f}%"))
            lines.append(box_line(f"Loss        : {snap.loss:.4f}"))
            lines.append(box_line(f"Best WER    : {snap.best_wer*100:.2f}%  (step {snap.best_wer_step})"))
            lines.append(box_line(f"Eval time   : {format_duration(snap.elapsed_sec)}  Samples: {snap.samples:,}"))
            lines.append(box_inner_bottom())

        # ---------- Dataset Statistics ----------
        if self._batches_this_window:
            avg_pad = self._pad_ratio_sum / max(1, self._batches_this_window)
            lines.append(box_inner_top("Dataset Statistics (since last report)"))
            lines.append(box_line(f"Batches     : {self._batches_this_window}"))
            lines.append(box_line(f"Frames      : {self._frames_this_window:,}"))
            lines.append(box_line(f"Tokens      : {self._tokens_this_window:,}"))
            lines.append(box_line(f"Input  len  : avg={self._input_len_avg:.0f}  max={self._input_len_max}"))
            lines.append(box_line(f"Target len  : avg={self._target_len_avg:.1f}  max={self._target_len_max}"))
            lines.append(box_line(f"Pad ratio   : {avg_pad*100:.2f}%"))
            lines.append(box_inner_bottom())

        lines.append(box_bottom())
        text = "\n".join(lines)
        for ln in lines:
            logger.info(ln)

        # Reset per-window counters
        self._batches_this_window = 0
        self._frames_this_window = 0
        self._tokens_this_window = 0
        self._input_len_avg = 0.0
        self._input_len_max = 0
        self._target_len_avg = 0.0
        self._target_len_max = 0
        self._pad_ratio_sum = 0.0

        return text

    # =====================================================================
    # Internals
    # =====================================================================

    def _loss_trend(self) -> tuple[str, float]:
        if len(self.loss_history) < 20:
            return ("? insufficient data", 0.0)
        recent = [l for _, l in list(self.loss_history)[-20:]]
        older = [l for _, l in list(self.loss_history)[-40:-20]] if len(self.loss_history) >= 40 else recent
        a = sum(recent) / len(recent)
        b = sum(older) / len(older)
        pct = 100.0 * (a - b) / max(1e-9, abs(b))
        if pct < -1:
            return ("↓ improving", pct)
        if pct < 1:
            return ("→ plateau", pct)
        return ("↑ worsening", pct)

    def _loss_variance(self) -> float:
        if len(self.loss_history) < 2:
            return 0.0
        recent = [l for _, l in list(self.loss_history)[-100:]]
        m = sum(recent) / len(recent)
        return sum((x - m) ** 2 for x in recent) / max(1, len(recent) - 1)

    def _convergence_rate(self) -> float:
        if len(self.loss_history) < 20:
            return 0.0
        a_idx, a_loss = self.loss_history[-20]
        b_idx, b_loss = self.loss_history[-1]
        if b_idx == a_idx:
            return 0.0
        # change in loss per 1K steps
        return 1000.0 * (b_loss - a_loss) / (b_idx - a_idx)

    def _gpu_memory(self) -> Optional[dict]:
        if not torch.cuda.is_available():
            return None
        try:
            dev = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(dev)
            allocated = torch.cuda.memory_allocated(dev)
            reserved = torch.cuda.memory_reserved(dev)
            peak = torch.cuda.max_memory_allocated(dev)
            return {
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": reserved / (1024**3),
                "total_gb": total / (1024**3),
                "peak_gb": peak / (1024**3),
                "frac": allocated / max(1, total),
            }
        except Exception:
            return None

    def _eta(self, step: int, max_steps: int, elapsed: float) -> float:
        if step <= 0:
            return float("inf")
        remaining_steps = max(0, max_steps - step)
        sps = step / max(1.0, elapsed)
        if sps <= 0:
            return float("inf")
        return remaining_steps / sps


