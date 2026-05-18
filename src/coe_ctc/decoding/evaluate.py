"""Evaluation entry point.

Decodes one or more LibriSpeech splits (test-clean / test-other / dev-clean /
dev-other), reports WER, CER, latency (ms/utt, RTF), and dumps REF/HYP files.

Invoked from ``scripts/evaluation/run_evaluation.sh``. Mirrors the CLI shape
of tsm-trainer008_aed's ``run_evaluation.sh`` (--models, --config, --beam,
--ngram, --checkpoint-avg).

Output table format:
    test_clean : WER  2.05 %   CER 0.61 %   latency 23.4 ms/utt  RTF 0.0041
    test_other : WER  4.72 %   CER 1.55 %   latency 24.1 ms/utt  RTF 0.0042
    dev_clean  : WER  2.01 %   CER 0.59 %   latency 23.2 ms/utt  RTF 0.0041
    dev_other  : WER  4.65 %   CER 1.49 %   latency 24.0 ms/utt  RTF 0.0042
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch

# Path bootstrap so the file works as ``python -m coe_ctc.decoding.evaluate``
# and also as ``python src/coe_ctc/decoding/evaluate.py``.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from coe_ctc.data.datamodule import CtcDataModule, DataModuleConfig
from coe_ctc.data.transforms import PerUtteranceMVN
from coe_ctc.decoding.beam import BeamDecoder, load_unigrams
from coe_ctc.decoding.checkpoint_average import ensure_averaged_checkpoint, select_top_n
from coe_ctc.decoding.greedy import greedy_decode_to_text
from coe_ctc.decoding.wer import compute_metrics, dump_refs_hyps
from coe_ctc.models.builder import build_model
from coe_ctc.utils.amp import pick_amp_dtype
from coe_ctc.utils.config import load_yaml, merge_overrides, resolve_paths
from coe_ctc.utils.logging import (
    plain_box_bottom,
    plain_box_line,
    plain_box_separator,
    plain_box_top,
    setup_logger,
)


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="coe_ctc.decoding.evaluate", description="CTC evaluation runner.")
    p.add_argument("--models", required=True, help="Path to best_checkpoints/ dir OR a single .pt file.")
    p.add_argument("--config", required=True, help="Evaluation YAML (e.g. scripts/evaluation/configs/libri.yaml).")
    p.add_argument("--set", action="append", default=[], help="Inline override key=value.")
    p.add_argument("--data", default=None, help="Override the dataset alias from config (libri100/libri960/libri_light).")
    p.add_argument("--beam", type=int, default=1, help="Beam size (default 1 = greedy).")
    p.add_argument("--ngram", default=None, help="Path to KenLM ARPA/binary for rescoring.")
    p.add_argument(
        "--lexicon",
        default=None,
        help=(
            "Path to a word list (librispeech-vocab.txt or librispeech-lexicon.txt) "
            "used as `unigrams=` for pyctcdecode. Restricts decoding to a closed "
            "vocabulary and improves WER on OOV-heavy LMs."
        ),
    )
    p.add_argument("--alpha", type=float, default=0.5, help="LM weight (default 0.5).")
    p.add_argument("--beta", type=float, default=1.5, help="Word-bonus (default 1.5).")
    p.add_argument("--checkpoint-avg", type=int, default=1, help="Average top-N best by WER (default 1 = off).")
    p.add_argument("--output-dir", default=None, help="Where to write CSV/REF/HYP outputs.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None, help="cpu / cuda / cuda:0 (default: auto).")
    p.add_argument("--amp", default="auto", choices=["auto", "bfloat16", "float16", "off"])
    p.add_argument("--max-batches", type=int, default=None, help="Truncate each split for smoke tests.")
    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────
# Per-split decode
# ─────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def decode_split(
    model: torch.nn.Module,
    loader,
    *,
    tokenizer,
    vocab: list[str],
    blank_idx: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    beam_size: int = 1,
    beam_decoder: Optional[BeamDecoder] = None,
    mvn: Optional[PerUtteranceMVN] = None,
    max_batches: Optional[int] = None,
) -> dict[str, Any]:
    """Run one split through the model and return refs, hyps, timings."""
    refs: list[str] = []
    hyps: list[str] = []
    cut_ids: list[str] = []
    audio_seconds = 0.0
    inference_seconds = 0.0
    n_utts = 0

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        features = batch["features"].to(device, non_blocking=True)
        feature_lengths = batch["feature_lengths"].to(device, non_blocking=True)
        if mvn is not None:
            features = mvn(features, feature_lengths)

        # 10 ms hop assumed (matches FbankConfig.frame_shift = 0.01s).
        audio_seconds += float(feature_lengths.sum().item()) * 0.01

        t0 = time.time()
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(features, feature_lengths)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds += time.time() - t0

        if beam_size > 1 and beam_decoder is not None:
            batch_hyps = beam_decoder.decode_batch(out.log_probs.float(), out.encoded_lens)
        else:
            batch_hyps = greedy_decode_to_text(out.log_probs.float(), out.encoded_lens, tokenizer, blank_idx=blank_idx)

        for i, hyp in enumerate(batch_hyps):
            refs.append(batch["texts"][i])
            hyps.append(hyp)
            cut_ids.append(batch["cut_ids"][i])
        n_utts += len(batch_hyps)

    return {
        "refs": refs,
        "hyps": hyps,
        "cut_ids": cut_ids,
        "audio_seconds": audio_seconds,
        "inference_seconds": inference_seconds,
        "n_utts": n_utts,
    }


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    setup_logger("coe_ctc", level=logging.INFO)
    logger = logging.getLogger("coe_ctc.eval")

    cfg = load_yaml(args.config)
    cfg = merge_overrides(cfg, args.set)
    cfg = resolve_paths(cfg, _REPO_ROOT)
    data_alias = args.data or cfg.get("data_alias")
    if not data_alias:
        raise ValueError("Provide --data or set 'data_alias' in the evaluation YAML.")

    # ----- Device / AMP -----
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp, amp_dtype = pick_amp_dtype(args.amp, device)

    # ----- Resolve checkpoint path -----
    models_arg = Path(args.models)
    if models_arg.is_dir():
        if args.checkpoint_avg and args.checkpoint_avg > 1:
            ckpt_path = ensure_averaged_checkpoint(models_arg, args.checkpoint_avg)
        else:
            picks = select_top_n(models_arg, 1)
            if not picks:
                raise FileNotFoundError(f"No checkpoints in {models_arg}.")
            ckpt_path = picks[0]
    else:
        ckpt_path = models_arg
    logger.info(f"Loading checkpoint: {ckpt_path}")

    # ----- Inspect training config from checkpoint -----
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    train_cfg = state.get("config") or {}
    mcfg = train_cfg.get("model", {})
    arch = mcfg.get("arch", cfg.get("model_arch", "conformer"))
    size = mcfg.get("size", cfg.get("model_size", "small"))
    blank_idx = int(mcfg.get("blank_idx", 0))
    num_features = int(train_cfg.get("data", {}).get("num_features", 80))

    # ----- Build datamodule -----
    dcfg_dict = dict(train_cfg.get("data", {}))
    # Allow eval YAML to override manifest_dir / bpe_model / test_parts:
    for k, v in cfg.get("data", {}).items():
        dcfg_dict[k] = v
    dcfg_dict.setdefault("manifest_dir", str(_REPO_ROOT / "downloads" / "manifests" / data_alias))
    dcfg_dict.setdefault("feats_dir", str(_REPO_ROOT / "downloads" / "feats" / f"{data_alias}_fbank80"))
    dcfg_dict.setdefault("manifest_prefix", "librispeech" if "libri" in data_alias and "light" not in data_alias else "libri-light")
    # Force evaluation-friendly settings
    dcfg_dict["apply_spec_augment"] = False
    dcfg_dict["num_workers"] = args.num_workers
    dcfg_dict["max_duration"] = float(cfg.get("max_duration", 100.0))
    dm_cfg = DataModuleConfig(
        **{k: v for k, v in dcfg_dict.items() if k in DataModuleConfig.__dataclass_fields__}
    )
    dm = CtcDataModule(dm_cfg)
    assert dm.tokenizer is not None
    vocab_size = dm.tokenizer.vocab_size
    vocab = dm.tokenizer.vocab_list()

    # ----- Build model + load weights (reuse the dict loaded above) -----
    model = build_model(
        arch=arch,
        size=size,
        vocab_size=vocab_size,
        num_features=num_features,
        blank_idx=blank_idx,
    )
    sd = state.get("model", state)
    msg = model.load_state_dict(sd, strict=True)
    logger.info(f"  Load state OK: missing={getattr(msg, 'missing_keys', None)} unexpected={getattr(msg, 'unexpected_keys', None)}")
    # Release the in-memory state dict — the model now owns the tensors.
    del state, sd
    model.to(device)
    model.eval()

    mvn = PerUtteranceMVN().to(device) if dm_cfg.per_utt_mvn else None

    # ----- Beam decoder -----
    beam_decoder = None
    if args.beam > 1 or args.ngram is not None:
        unigrams = load_unigrams(args.lexicon)
        beam_decoder = BeamDecoder(
            vocab=vocab,
            beam_size=args.beam,
            blank_idx=blank_idx,
            ngram_path=args.ngram,
            unigrams=unigrams,
            alpha=args.alpha,
            beta=args.beta,
        )
        logger.info(
            f"Beam decoder: beam={args.beam}  ngram={'on' if args.ngram else 'off'}  "
            f"lexicon={'on (' + str(len(unigrams)) + ' words)' if unigrams else 'off'}"
        )
    else:
        logger.info("Decoder: greedy")

    # ----- Output dir -----
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.models).parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = _make_tag(args)

    # ----- Run all splits — reuse the same datamodule + tokenizer -----
    splits = cfg.get("splits") or dm_cfg.test_parts
    dm_cfg.test_parts = tuple(splits)
    loaders = dm.test_loaders()
    rows: list[dict] = []
    for part in splits:
        loader = loaders[part]
        logger.info(f"Decoding split '{part}' …")
        r = decode_split(
            model,
            loader,
            tokenizer=dm.tokenizer,
            vocab=vocab,
            blank_idx=blank_idx,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            beam_size=args.beam,
            beam_decoder=beam_decoder,
            mvn=mvn,
            max_batches=args.max_batches,
        )
        metrics = compute_metrics(r["refs"], r["hyps"])
        latency_ms = 1000.0 * r["inference_seconds"] / max(1, r["n_utts"])
        rtf = r["inference_seconds"] / max(1e-6, r["audio_seconds"])
        rows.append(
            {
                "split": part,
                "wer": metrics["wer"],
                "cer": metrics["cer"],
                "n_utts": r["n_utts"],
                "audio_sec": r["audio_seconds"],
                "inference_sec": r["inference_seconds"],
                "latency_ms": latency_ms,
                "rtf": rtf,
            }
        )
        # Dump REF/HYP file
        ref_hyp_path = output_dir / f"{tag}__{part.replace('-', '_')}.refhyp.txt"
        dump_refs_hyps(ref_hyp_path, r["refs"], r["hyps"], cut_ids=r["cut_ids"])
        logger.info(f"  → {ref_hyp_path}")

    # ----- Pretty print summary -----
    title = f"CTC EVALUATION — {tag}"
    lines = [plain_box_top(title)]
    header = f"{'Split':<11}  {'WER':>8}   {'CER':>8}   {'Latency':>14}   {'RTF':>10}"
    lines.append(plain_box_line(header))
    lines.append(plain_box_separator())
    for r in rows:
        row = (
            f"{r['split']:<11}  "
            f"{r['wer']*100:>7.2f}%   "
            f"{r['cer']*100:>7.2f}%   "
            f"{r['latency_ms']:>10.2f} ms/utt   "
            f"{r['rtf']:>10.4f}"
        )
        lines.append(plain_box_line(row))
    lines.append(plain_box_bottom())
    for ln in lines:
        logger.info(ln)

    # ----- Write CSV / JSON summary -----
    csv_path = output_dir / f"{tag}.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("split,wer,cer,n_utts,latency_ms,rtf,audio_sec,inference_sec\n")
        for r in rows:
            fh.write(
                f"{r['split']},{r['wer']:.6f},{r['cer']:.6f},{r['n_utts']},"
                f"{r['latency_ms']:.3f},{r['rtf']:.5f},{r['audio_sec']:.2f},{r['inference_sec']:.2f}\n"
            )
    logger.info(f"  → {csv_path}")
    (output_dir / f"{tag}.json").write_text(json.dumps(rows, indent=2))

    return 0


def _make_tag(args) -> str:
    parts = []
    parts.append(f"beam{args.beam}")
    if args.ngram:
        parts.append("ngram=" + Path(args.ngram).stem)
    if args.checkpoint_avg > 1:
        parts.append(f"avg{args.checkpoint_avg}")
    return "_".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
