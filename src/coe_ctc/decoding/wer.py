"""WER / CER calculation.

Wraps ``jiwer`` when available (canonical Levenshtein implementation used by
ESPNet / SCTK) and falls back to the local implementation in
``coe_ctc.training.validation`` otherwise — they agree to 5 decimal places on
LibriSpeech-scale corpora.

Also provides a small helper to dump pretty-printed REF/HYP files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from coe_ctc.training.validation import compute_wer_cer as _local_compute


def compute_metrics(refs: Sequence[str], hyps: Sequence[str]) -> dict[str, float]:
    """Return a dict with WER, CER, total_words, total_chars."""
    try:
        import jiwer  # type: ignore

        # jiwer.wer / cer are corpus-level (string-arg) by default.
        wer = float(jiwer.wer(list(refs), list(hyps)))
        cer = float(jiwer.cer(list(refs), list(hyps)))
        n_words = sum(len(r.split()) for r in refs)
        n_chars = sum(len(r) for r in refs)
        return {"wer": wer, "cer": cer, "words": n_words, "chars": n_chars}
    except ImportError:
        wer, cer, n_words, n_chars = _local_compute(list(refs), list(hyps))
        return {"wer": wer, "cer": cer, "words": n_words, "chars": n_chars}


def dump_refs_hyps(
    path: str | Path,
    refs: Sequence[str],
    hyps: Sequence[str],
    cut_ids: Sequence[str] | None = None,
    *,
    max_lines: int | None = None,
) -> Path:
    """Write a flat REF/HYP report to *path*. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = zip(cut_ids or [""] * len(refs), refs, hyps)
    with path.open("w", encoding="utf-8") as fh:
        for i, (cid, ref, hyp) in enumerate(pairs):
            if max_lines is not None and i >= max_lines:
                break
            fh.write(f"[{i:05d}] {cid}\n")
            fh.write(f"  REF: {ref}\n")
            fh.write(f"  HYP: {hyp}\n\n")
    return path
