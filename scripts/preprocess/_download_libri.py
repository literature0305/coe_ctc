#!/usr/bin/env python3
"""End-to-end LibriSpeech / libri-light preprocess entry point.

Invoked from ``scripts/preprocess/run_preprocess.sh``. Performs:

  1. Download + extract the raw corpus (skips if present).
  2. Build Lhotse recording/supervision manifests.
  3. Extract fBank-80 features → ``cuts_{part}.jsonl.gz``.
  4. Train one or more SentencePiece BPE models (in parallel).
  5. (libri960 only) Derive a libri100/ manifest dir from train-clean-100.

All steps honor ``--num-workers`` (default 8) as the upper bound on CPU
parallelism so a shared server isn't overwhelmed.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make the repo's `src/` importable when this script is invoked directly via
# ``python scripts/preprocess/_download_libri.py``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from coe_ctc.data import (  # noqa: E402  — runtime path manipulation above
    BpeTrainConfig,
    DATASETS,
    FbankConfig,
    compute_fbank,
    download_dataset,
    prepare_manifests,
    train_bpe_parallel,
)
from coe_ctc.data.librispeech import (  # noqa: E402
    LM_RESOURCES,
    download_lm_resources,
    dump_transcripts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_preprocess",
        description="Download + manifest + fBank + BPE preprocessing for LibriSpeech / libri-light.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data",
        required=True,
        choices=sorted(DATASETS.keys()),
        help="Dataset preset (libri100 / libri960 / libri_light).",
    )
    p.add_argument(
        "--n_bpe",
        type=int,
        nargs="+",
        default=[3000],
        help="One or more BPE vocab sizes to train in parallel. Example: --n_bpe 3000 300.",
    )
    p.add_argument(
        "--download-dir",
        default="downloads",
        help="Root for raw audio + manifests + features + bpe (default: ./downloads).",
    )
    p.add_argument("--num-workers", type=int, default=8, help="Max parallel CPU workers (default: 8).")
    p.add_argument("--num-mel-bins", type=int, default=80, help="fBank dim (default: 80).")
    p.add_argument("--sampling-rate", type=int, default=16000, help="(default: 16000).")
    p.add_argument(
        "--bpe-model-type",
        default="unigram",
        choices=["unigram", "bpe", "char"],
        help="SentencePiece model type (default: unigram).",
    )
    p.add_argument(
        "--libri-light-splits",
        nargs="+",
        default=None,
        help="Override libri-light splits to download (default: spec default = ['small']).",
    )
    p.add_argument(
        "--skip-download", action="store_true", help="Skip download/extract phase (assume corpus is in place)."
    )
    p.add_argument(
        "--skip-manifest", action="store_true", help="Skip manifest building (assume manifests exist)."
    )
    p.add_argument(
        "--skip-fbank", action="store_true", help="Skip fBank extraction (assume cuts_*.jsonl.gz exist)."
    )
    p.add_argument("--skip-bpe", action="store_true", help="Skip BPE training.")
    p.add_argument(
        "--with-lm",
        nargs="*",
        default=None,
        metavar="RESOURCE",
        help=(
            "Also download LibriSpeech LM resources from openslr-11. With no arguments, "
            "downloads the recommended bundle (vocab + 3-gram pruned + 4-gram). "
            "Pass 'all' to also fetch the lexicon and 4G corpus, or list specific "
            f"resources: {sorted(LM_RESOURCES)}."
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print what would happen without touching disk."
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Cap thread libraries — same pattern as tsm-trainer's train.sh.
    nw = max(1, int(args.num_workers))
    for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        os.environ.setdefault(env_var, str(nw))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("preprocess")

    download_root = Path(args.download_dir).resolve()
    spec = DATASETS[args.data]

    # ----- prefix used in manifest filenames is fixed per-source -----
    manifest_prefix = "librispeech" if spec.source == "librispeech" else "libri-light"

    # ----- per-dataset directories -----
    manifest_dir = download_root / "manifests" / args.data
    feats_dir = download_root / "feats" / f"{args.data}_fbank{args.num_mel_bins}"
    bpe_root = download_root / "bpe"
    transcripts_dir = download_root / "transcripts"

    log.info("=" * 78)
    log.info(f"  Preprocess: {args.data}")
    log.info(f"  Download dir   : {download_root}")
    log.info(f"  Manifest dir   : {manifest_dir}")
    log.info(f"  Features dir   : {feats_dir}")
    log.info(f"  BPE root       : {bpe_root}")
    log.info(f"  BPE vocab sizes: {args.n_bpe}")
    log.info(f"  num_workers    : {nw}")
    log.info(f"  Source         : {spec.source}")
    log.info(f"  Splits (train) : {spec.train_splits}")
    log.info(f"  Splits (dev)   : {spec.dev_splits}")
    log.info(f"  Splits (test)  : {spec.test_splits}")
    log.info("=" * 78)

    if args.dry_run:
        log.info("[dry-run] exiting before any side-effects.")
        return 0

    # ─────────────── 1. Download ───────────────
    if args.skip_download:
        log.info("[skip-download] honoring --skip-download.")
        corpus_dir = download_root / "corpora" / ("LibriSpeech" if spec.source == "librispeech" else "libri_light")
    else:
        corpus_dir = download_dataset(
            name=args.data,
            download_dir=download_root,
            libri_light_splits=args.libri_light_splits,
            log=log,
        )

    # ─────────────── 2. Manifests ───────────────
    if args.skip_manifest:
        log.info("[skip-manifest] honoring --skip-manifest.")
    else:
        prepare_manifests(
            name=args.data,
            corpus_dir=corpus_dir,
            manifest_dir=manifest_dir,
            num_workers=nw,
            log=log,
        )

    # ─────────────── 3. fBank features ───────────────
    if args.skip_fbank:
        log.info("[skip-fbank] honoring --skip-fbank.")
    else:
        fbank_cfg = FbankConfig(num_mel_bins=args.num_mel_bins, sampling_rate=args.sampling_rate)
        compute_fbank(
            manifest_dir=manifest_dir,
            feats_dir=feats_dir,
            num_workers=nw,
            config=fbank_cfg,
            prefix=manifest_prefix,
            log=log,
        )
        # If we prepped 960h, also extract fBank into the derived 100h manifests
        # (which symlink the same recordings + supervisions).
        if args.data == "libri960":
            derived = download_root / "manifests" / "libri100"
            if derived.exists():
                compute_fbank(
                    manifest_dir=derived,
                    feats_dir=download_root / "feats" / f"libri100_fbank{args.num_mel_bins}",
                    num_workers=nw,
                    config=fbank_cfg,
                    prefix=manifest_prefix,
                    log=log,
                )

    # ─────────────── 4. BPE ───────────────
    if args.skip_bpe:
        log.info("[skip-bpe] honoring --skip-bpe.")
    else:
        # Pick the supervisions we feed to SentencePiece. For libri960 we use
        # the FULL 960h transcripts so the same BPE works for both libri100
        # and libri960. For libri100 we use train-clean-100 only.
        sup_paths: list[Path] = []
        if spec.source == "librispeech":
            train_parts = spec.train_splits if args.data != "libri100" else ("train-clean-100",)
            for part in train_parts:
                sup_path = manifest_dir / f"librispeech_supervisions_{part}.jsonl.gz"
                if sup_path.exists():
                    sup_paths.append(sup_path)
                else:
                    log.warning(f"  [bpe] missing supervisions for {part}, skipping.")
        elif spec.source == "libri_light":
            # libri-light is unlabeled — we cannot train BPE from it. Reuse libri960's.
            log.info(
                "  [bpe] libri-light is unlabeled; skipping BPE training. "
                "Use the libri960 BPE model when training on libri_light."
            )
            sup_paths = []

        if sup_paths:
            transcripts_dir.mkdir(parents=True, exist_ok=True)
            txt_path = transcripts_dir / f"{args.data}_transcripts.txt"
            if not txt_path.exists():
                # Preserve case: datamodule.py feeds raw upper-case supervision
                # text to SentencePiece, so BPE must be trained on the same.
                dump_transcripts(sup_paths, txt_path, lowercase=False, log=log)
            else:
                log.info(f"  [skip-transcripts] {txt_path} already exists.")

            train_bpe_parallel(
                transcript_file=txt_path,
                output_root=bpe_root,
                vocab_sizes=args.n_bpe,
                dataset_name=args.data,
                num_workers=nw,
                base_config=BpeTrainConfig(model_type=args.bpe_model_type),
                log=log,
            )

            # Mirror libri960's BPE under libri100/ as well so either alias works.
            if args.data == "libri960":
                for V in args.n_bpe:
                    src = bpe_root / f"libri960_bpe{V}"
                    dst = bpe_root / f"libri100_bpe{V}"
                    if src.exists() and not dst.exists():
                        try:
                            dst.symlink_to(src.resolve())
                            log.info(f"  Mirrored BPE: {dst} → {src}")
                        except (OSError, NotImplementedError):
                            log.warning(f"  symlink failed for {dst}; users must train libri100 BPE separately.")

    # ─────────────── 5. LM resources (optional) ───────────────
    if args.with_lm is not None:
        items = args.with_lm if args.with_lm else None  # empty list → use defaults
        log.info(f"Downloading LM resources from openslr-11 (items={items or 'default bundle'}).")
        paths = download_lm_resources(download_root, items=items, log=log)
        for key, path in paths.items():
            log.info(f"  {key:14s} → {path}")

    log.info("Preprocess complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
