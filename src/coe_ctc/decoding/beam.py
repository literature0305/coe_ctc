"""CTC prefix beam search.

Two backends:
    1. ``pyctcdecode`` (preferred, fast, supports KenLM rescoring).
    2. Pure-Python fallback (no LM, used only when pyctcdecode is unavailable).

Both backends operate on **log-probabilities** of shape ``(T, V)`` per sample
and emit final hypotheses as plain text (BPE pieces already collapsed by the
tokenizer).
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Sequence

import torch

# ``load_unigrams`` lives next to the LM-resource downloader in
# ``coe_ctc.data.librispeech`` since it parses files produced by that module.
# Re-export here for backwards compatibility with the public decoding API.
from coe_ctc.data.librispeech import load_unigrams  # noqa: F401

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# pyctcdecode backend
# ─────────────────────────────────────────────────────────────────────────


class BeamDecoder:
    """Wraps either pyctcdecode or the local fallback decoder.

    Args:
        vocab: Token strings indexed by id (blank goes through anyway).
        beam_size: Beam width passed to pyctcdecode.
        blank_idx: CTC blank label id (default 0).
        ngram_path: Optional KenLM ARPA / binary for shallow-fusion rescoring.
            ``alpha`` / ``beta`` are only consulted when this is set.
        unigrams: Optional closed-vocabulary word list (e.g. from
            ``load_unigrams("librispeech-vocab.txt")``). When supplied,
            pyctcdecode restricts hypotheses to this vocabulary; greatly
            improves WER on OOV-heavy LMs at a small speed cost.
        alpha: KenLM weight in the log-linear score combination.
        beta: Word-bonus term that counteracts the LM's bias toward shorter
            transcripts.
        prefer_pyctcdecode: If False, skip pyctcdecode entirely and use the
            local LM-free prefix beam (useful for debugging).
    """

    def __init__(
        self,
        *,
        vocab: Sequence[str],
        beam_size: int = 4,
        blank_idx: int = 0,
        ngram_path: Optional[str] = None,
        unigrams: Optional[Sequence[str]] = None,
        alpha: float = 0.5,
        beta: float = 1.5,
        prefer_pyctcdecode: bool = True,
    ) -> None:
        self.vocab = list(vocab)
        self.beam_size = int(beam_size)
        self.blank_idx = int(blank_idx)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.ngram_path = ngram_path
        # Restricts pyctcdecode to a closed word vocabulary — typically the
        # librispeech-vocab.txt or librispeech-lexicon.txt word list. Greatly
        # improves WER on OOV-heavy LMs at the cost of slightly slower decoding.
        self.unigrams = list(unigrams) if unigrams is not None else None
        self._pyctc = None

        if prefer_pyctcdecode:
            self._pyctc = self._try_pyctcdecode()
            if self._pyctc is None and ngram_path is not None:
                logger.warning(
                    "pyctcdecode not available; falling back to LM-free pure-Python beam decoder. "
                    "Install pyctcdecode (and kenlm) to enable N-gram rescoring."
                )

    def _try_pyctcdecode(self):
        try:
            from pyctcdecode import build_ctcdecoder
        except ImportError:
            return None

        # pyctcdecode expects labels with the blank as the empty string.
        labels = [tok if i != self.blank_idx else "" for i, tok in enumerate(self.vocab)]
        try:
            decoder = build_ctcdecoder(
                labels,
                kenlm_model_path=self.ngram_path,
                alpha=self.alpha,
                beta=self.beta,
                unigrams=self.unigrams,
            )
        except Exception as exc:
            logger.warning(f"pyctcdecode build failed: {exc}; falling back to local decoder.")
            return None
        return decoder

    # --------------------------------------------------------------- batch API
    def decode_batch(
        self,
        log_probs: torch.Tensor,
        lengths: torch.Tensor,
    ) -> List[str]:
        """Decode a (B, T, V) tensor → list of B text hypotheses."""
        if self._pyctc is not None:
            return self._decode_pyctc(log_probs, lengths)
        return self._decode_local(log_probs, lengths)

    # ---------------------------------------------------------- pyctcdecode
    def _decode_pyctc(self, log_probs: torch.Tensor, lengths: torch.Tensor) -> List[str]:
        lp = log_probs.detach().cpu().float().numpy()
        out: List[str] = []
        for i, L in enumerate(lengths.tolist()):
            arr = lp[i, : int(L)]
            text = self._pyctc.decode(arr, beam_width=self.beam_size)
            out.append(text)
        return out

    # ------------------------------------------------- pure-Python (no LM)
    def _decode_local(self, log_probs: torch.Tensor, lengths: torch.Tensor) -> List[str]:
        out: List[str] = []
        lp = log_probs.detach().float()
        for i, L in enumerate(lengths.tolist()):
            sample = lp[i, : int(L)]
            hyp_ids = _prefix_beam_search(sample, beam_size=self.beam_size, blank_idx=self.blank_idx)
            text = "".join(self.vocab[j] for j in hyp_ids).replace("▁", " ").strip()
            out.append(text)
        return out


# ─────────────────────────────────────────────────────────────────────────
# Local prefix beam search (LM-free)
# ─────────────────────────────────────────────────────────────────────────


def _logsumexp(*vals: float) -> float:
    """Stable log-sum-exp for a small number of arguments."""
    m = max(vals)
    if m == float("-inf"):
        return m
    return m + math.log(sum(math.exp(v - m) for v in vals))


def _prefix_beam_search(
    log_probs: torch.Tensor,
    *,
    beam_size: int = 4,
    blank_idx: int = 0,
) -> list[int]:
    """Standard prefix beam search (Graves & Jaitly, 2014).

    For LibriSpeech-scale vocabularies this is slow in pure Python; we rely on
    pyctcdecode in practice and keep this only as a fallback. Returns the
    best beam as a list of vocab indices.

    The state for each beam is ``(prefix, log_pb, log_pnb)``:
        log_pb  = log-prob ending in blank
        log_pnb = log-prob ending in non-blank
    """
    T, V = log_probs.shape
    # Initial beam: empty prefix, all probability mass on "ending in blank".
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, float("-inf"))}

    for t in range(T):
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}
        lp_t = log_probs[t].tolist()

        for prefix, (pb, pnb) in beams.items():
            last = prefix[-1] if prefix else -1
            for v in range(V):
                lp = lp_t[v]
                if v == blank_idx:
                    new_pb = _logsumexp(
                        next_beams.get(prefix, (float("-inf"), float("-inf")))[0],
                        pb + lp,
                        pnb + lp,
                    )
                    new_pnb = next_beams.get(prefix, (float("-inf"), float("-inf")))[1]
                    next_beams[prefix] = (new_pb, new_pnb)
                elif v == last:
                    # Repeated non-blank: extends same prefix only via blank-ending branch.
                    new_pnb_self = _logsumexp(
                        next_beams.get(prefix, (float("-inf"), float("-inf")))[1],
                        pnb + lp,
                    )
                    next_beams[prefix] = (
                        next_beams.get(prefix, (float("-inf"), float("-inf")))[0],
                        new_pnb_self,
                    )
                    new_prefix = prefix + (v,)
                    cur = next_beams.get(new_prefix, (float("-inf"), float("-inf")))
                    new_pnb_ext = _logsumexp(cur[1], pb + lp)
                    next_beams[new_prefix] = (cur[0], new_pnb_ext)
                else:
                    new_prefix = prefix + (v,)
                    cur = next_beams.get(new_prefix, (float("-inf"), float("-inf")))
                    new_pnb_ext = _logsumexp(cur[1], pb + lp, pnb + lp)
                    next_beams[new_prefix] = (cur[0], new_pnb_ext)

        # Prune to beam_size by total log-prob.
        scored = sorted(
            next_beams.items(),
            key=lambda kv: _logsumexp(kv[1][0], kv[1][1]),
            reverse=True,
        )
        beams = dict(scored[:beam_size])

    best = max(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))
    return list(best[0])
