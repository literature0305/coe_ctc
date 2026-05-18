"""Per-epoch validation on dev-other.

Produces the BENCHMARK EVALUATION box, REF/HYP samples, TensorBoard scalars,
and a :class:`BenchmarkSnapshot` returned to the health reporter.

Decoding here is greedy only (argmax + collapse-blank) for speed. Full beam
+ N-gram rescoring lives in Phase 4's evaluation script.
"""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from coe_ctc.training.health import BenchmarkSnapshot
from coe_ctc.utils.logging import (
    format_duration,
    plain_box_bottom,
    plain_box_line,
    plain_box_separator,
    plain_box_top,
)


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Lightweight greedy decoder
# ─────────────────────────────────────────────────────────────────────────


def ctc_greedy_decode(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    blank_idx: int = 0,
) -> list[list[int]]:
    """Argmax + collapse-blank decode.

    Args:
        log_probs: (B, T, V)
        lengths:   (B,) — valid time steps per sample
        blank_idx: blank label id (default 0).

    Returns:
        List of decoded id sequences (no blanks, no repeats).
    """
    argmax = log_probs.argmax(dim=-1)
    out: list[list[int]] = []
    for i, L in enumerate(lengths.tolist()):
        ids = argmax[i, : int(L)].tolist()
        prev = -1
        collapsed: list[int] = []
        for tok in ids:
            if tok != prev:
                if tok != blank_idx:
                    collapsed.append(int(tok))
                prev = tok
        out.append(collapsed)
    return out


# ─────────────────────────────────────────────────────────────────────────
# WER / CER (jiwer-free local implementation; jiwer used in Phase 4 for parity)
# ─────────────────────────────────────────────────────────────────────────


def _edit_distance(ref: list, hyp: list) -> int:
    """Standard Levenshtein DP — used by both WER (tokens) and CER (chars)."""
    nr, nh = len(ref), len(hyp)
    if nr == 0:
        return nh
    if nh == 0:
        return nr
    prev = list(range(nh + 1))
    cur = [0] * (nh + 1)
    for i in range(1, nr + 1):
        cur[0] = i
        for j in range(1, nh + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[nh]


def compute_wer_cer(refs: list[str], hyps: list[str]) -> tuple[float, float, int, int]:
    """Returns (WER, CER, total_words, total_chars)."""
    total_w_err = 0
    total_w = 0
    total_c_err = 0
    total_c = 0
    for ref, hyp in zip(refs, hyps):
        rw = ref.strip().split()
        hw = hyp.strip().split()
        total_w_err += _edit_distance(rw, hw)
        total_w += max(1, len(rw))
        rc = list(ref.strip())
        hc = list(hyp.strip())
        total_c_err += _edit_distance(rc, hc)
        total_c += max(1, len(rc))
    return total_w_err / total_w, total_c_err / total_c, total_w, total_c


# ─────────────────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────────────────


class Validator:
    """Run validation on dev-other (or any other split) and pretty-print results."""

    def __init__(
        self,
        *,
        tokenizer,
        blank_idx: int = 0,
        sample_count: int = 10,
        output_dir: Optional[str | Path] = None,
        device: torch.device | str = "cuda",
    ) -> None:
        self.tokenizer = tokenizer
        self.blank_idx = blank_idx
        self.sample_count = sample_count
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.device = torch.device(device) if isinstance(device, str) else device

        # Cumulative tracking across epochs — best values are derived from history.
        self.history: list[dict] = []

    @property
    def best_wer(self) -> float:
        return min((r["wer"] for r in self.history), default=float("inf"))

    @property
    def best_wer_step(self) -> int:
        if not self.history:
            return 0
        return min(self.history, key=lambda r: r["wer"])["step"]

    @property
    def best_cer(self) -> float:
        return min((r["cer"] for r in self.history), default=float("inf"))

    @property
    def best_cer_step(self) -> int:
        if not self.history:
            return 0
        return min(self.history, key=lambda r: r["cer"])["step"]

    # ----------------------------------------------------------------- core
    @torch.no_grad()
    def run(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        *,
        step: int,
        epoch: int,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
        max_batches: Optional[int] = None,
    ) -> BenchmarkSnapshot:
        """Decode the full loader, compute WER/CER/loss, render results.

        For CoE models the model returns a ``CoeOutput`` whose ``per_pass``
        carries one ``CTCOutput`` per encoder. Every pass is scored; the
        reported / checkpointed metric is always the last pass (m = M-1).
        """
        was_training = model.training
        model.eval()
        t0 = time.time()

        # Per-pass accumulators. For non-CoE the list has length 1.
        num_passes: Optional[int] = None
        per_pass_loss: list[float] = []
        per_pass_loss_w: list[int] = []
        per_pass_refs: list[list[str]] = []
        per_pass_hyps: list[list[str]] = []
        per_pass_samples: list[list[tuple[str, str]]] = []
        sample_indices = set(range(self.sample_count))

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            features = batch["features"].to(self.device, non_blocking=True)
            feature_lengths = batch["feature_lengths"].to(self.device, non_blocking=True)
            tokens = batch["tokens"].to(self.device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(self.device, non_blocking=True)

            ctx = (
                torch.amp.autocast(device_type=self.device.type, dtype=amp_dtype)
                if use_amp
                else contextlib.nullcontext()
            )
            with ctx:
                out = model(features, feature_lengths, targets=tokens, target_lengths=target_lengths)

            passes = getattr(out, "per_pass", None) or [out]
            if num_passes is None:
                num_passes = len(passes)
                per_pass_loss = [0.0] * num_passes
                per_pass_loss_w = [0] * num_passes
                per_pass_refs = [[] for _ in range(num_passes)]
                per_pass_hyps = [[] for _ in range(num_passes)]
                per_pass_samples = [[] for _ in range(num_passes)]

            for m, p_out in enumerate(passes):
                if p_out.loss is not None and torch.isfinite(p_out.loss):
                    per_pass_loss[m] += p_out.loss.item() * features.size(0)
                    per_pass_loss_w[m] += features.size(0)
                decoded_ids = ctc_greedy_decode(
                    p_out.log_probs.float(), p_out.encoded_lens, blank_idx=self.blank_idx
                )
                for i, ids in enumerate(decoded_ids):
                    ref = batch["texts"][i]
                    hyp = self.tokenizer.decode(ids) if ids else ""
                    per_pass_refs[m].append(ref)
                    per_pass_hyps[m].append(hyp)
                    if len(per_pass_samples[m]) < self.sample_count or batch_idx in sample_indices:
                        per_pass_samples[m].append((ref, hyp))
                        per_pass_samples[m] = per_pass_samples[m][: self.sample_count]

        elapsed = time.time() - t0
        assert num_passes is not None, "Validation loader yielded zero batches."

        per_pass_rows: list[dict] = []
        for m in range(num_passes):
            avg_loss_m = per_pass_loss[m] / max(1, per_pass_loss_w[m])
            wer_m, cer_m, nw_m, nc_m = compute_wer_cer(per_pass_refs[m], per_pass_hyps[m])
            per_pass_rows.append(
                {
                    "m": m,
                    "wer": wer_m,
                    "cer": cer_m,
                    "loss": avg_loss_m,
                    "samples": len(per_pass_refs[m]),
                    "words": nw_m,
                    "chars": nc_m,
                }
            )

        # The "primary" row is the last pass (m=M-1).
        primary = per_pass_rows[-1]
        row = {
            "step": step,
            "epoch": epoch,
            "wer": primary["wer"],
            "cer": primary["cer"],
            "loss": primary["loss"],
            "samples": primary["samples"],
            "words": primary["words"],
            "chars": primary["chars"],
            "elapsed_sec": elapsed,
            "per_pass": per_pass_rows,
        }
        self.history.append(row)

        # ---- BENCHMARK EVALUATION pretty box ----
        self._render_box(row, samples=per_pass_samples)

        # ---- REF/HYP file ----
        if self.output_dir is not None:
            self._dump_samples(step=step, epoch=epoch, samples=per_pass_samples)

        if was_training:
            model.train()

        return BenchmarkSnapshot(
            step=step,
            wer=primary["wer"],
            cer=primary["cer"],
            loss=primary["loss"],
            best_wer=self.best_wer,
            best_wer_step=self.best_wer_step,
            elapsed_sec=elapsed,
            samples=primary["samples"],
        )

    # ----------------------------------------------------------------- pretty
    def _render_box(
        self,
        row: dict,
        samples: list[list[tuple[str, str]]],
    ) -> None:
        title = f"BENCHMARK EVALUATION — dev-other — Step {row['step']} (epoch {row['epoch']})"
        lines = [plain_box_top(title)]
        lines.append(plain_box_line(""))
        per_pass = row.get("per_pass") or [
            {"m": 0, "wer": row["wer"], "cer": row["cer"], "loss": row["loss"]}
        ]
        is_coe = len(per_pass) > 1
        last_m = per_pass[-1]["m"]
        if is_coe:
            lines.append(plain_box_line(f"Passes       : {len(per_pass)}  (M-th = best-ckpt target)"))
            for r in per_pass:
                tag = " *" if r["m"] == last_m else "  "
                lines.append(
                    plain_box_line(
                        f"m={r['m']:>2d}{tag}    WER {r['wer']*100:6.2f}%   "
                        f"CER {r['cer']*100:6.2f}%   loss {r['loss']:7.4f}"
                    )
                )
            lines.append(plain_box_separator())
        lines.append(
            plain_box_line(
                f"WER          : {row['wer']*100:.2f}%   "
                f"(best: {self.best_wer*100:.2f}% @ step {self.best_wer_step})"
            )
        )
        lines.append(
            plain_box_line(
                f"CER          : {row['cer']*100:.2f}%   "
                f"(best: {self.best_cer*100:.2f}% @ step {self.best_cer_step})"
            )
        )
        lines.append(plain_box_line(f"Loss         : {row['loss']:.4f}"))
        lines.append(plain_box_line(f"Samples      : {row['samples']:,}   Words: {row['words']:,}   Chars: {row['chars']:,}"))
        lines.append(plain_box_line(f"Elapsed      : {format_duration(row['elapsed_sec'])}"))
        # REF/HYP samples — for CoE show first + last passes; non-CoE just the only set.
        show_indices = [0, len(samples) - 1] if is_coe and len(samples) > 1 else [len(samples) - 1]
        for sm_idx in show_indices:
            sample_set = samples[sm_idx]
            lines.append(plain_box_separator())
            lines.append(plain_box_line(f"REF / HYP samples — m={sm_idx} (showing {len(sample_set)})"))
            lines.append(plain_box_separator())
            for i, (ref, hyp) in enumerate(sample_set, start=1):
                lines.append(plain_box_line(f"[{i:02d}] REF: {ref}"))
                lines.append(plain_box_line(f"     HYP: {hyp}"))
        lines.append(plain_box_bottom())
        for ln in lines:
            logger.info(ln)

    def _dump_samples(
        self,
        *,
        step: int,
        epoch: int,
        samples: list[list[tuple[str, str]]],
    ) -> None:
        assert self.output_dir is not None
        target = self.output_dir / f"valid_samples_epoch{epoch}_step{step}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            fh.write(f"# step={step} epoch={epoch}\n")
            for m_idx, sample_set in enumerate(samples):
                fh.write(f"\n# ---- m={m_idx} ----\n")
                for i, (ref, hyp) in enumerate(sample_set, start=1):
                    fh.write(f"[{i:02d}] REF: {ref}\n")
                    fh.write(f"     HYP: {hyp}\n\n")


