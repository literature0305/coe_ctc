"""KenLM N-gram language model training wrapper.

The heavy work is done by the standalone ``lmplz`` binary (compiled from
KenLM, https://github.com/kpu/kenlm). This script:

  1. Loads a YAML config describing the model order, pruning, and output paths.
  2. Decompresses gzipped corpora on-the-fly so the user can point at
     ``librispeech-lm-norm.txt.gz`` directly.
  3. Pipes the corpus into ``lmplz`` and converts the ARPA to a binary KenLM
     ``.klm`` if ``build_binary`` is available.

Both ``lmplz`` and ``build_binary`` must be on ``$PATH``. Install via:

    sudo apt-get install -y libboost-all-dev cmake
    git clone https://github.com/kpu/kenlm.git
    cd kenlm && mkdir build && cd build && cmake .. && make -j

CLI usage:
    python -m coe_ctc.lm.train_ngram \
        --config scripts/training/configs/4gram.yaml \
        --corpus downloads/corpus/librispeech-lm-norm.txt.gz
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Path bootstrap
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from coe_ctc.utils.config import load_yaml, merge_overrides, resolve_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="coe_ctc.lm.train_ngram",
                                description="Train an N-gram LM with KenLM lmplz.")
    p.add_argument("--config", required=True, help="YAML config (order, pruning, output_path).")
    p.add_argument("--corpus", required=True, help="One-sentence-per-line corpus (.txt or .txt.gz).")
    p.add_argument("--set", action="append", default=[], help="Inline override key=value.")
    p.add_argument("--lmplz", default="lmplz", help="Path to KenLM lmplz binary (default: from PATH).")
    p.add_argument("--build-binary", default="build_binary",
                   help="Path to KenLM build_binary (default: from PATH).")
    p.add_argument("--dry-run", action="store_true", help="Print the lmplz command and exit.")
    return p.parse_args(argv)


def _decompress_if_needed(corpus: Path, tmp_dir: Path) -> Path:
    if corpus.suffix == ".gz":
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out = tmp_dir / corpus.stem
        if out.exists():
            return out
        logging.info(f"  Decompressing {corpus} → {out} …")
        with gzip.open(corpus, "rt", encoding="utf-8") as src, open(out, "w", encoding="utf-8") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
        return out
    return corpus


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    cfg = load_yaml(args.config)
    cfg = merge_overrides(cfg, args.set)
    cfg = resolve_paths(cfg, _REPO_ROOT)

    order = int(cfg.get("order", 4))
    prune = list(cfg.get("prune", [0, 0, 1, 1]))      # ignore singletons in 3- and 4-gram tiers
    out_dir = Path(cfg.get("output_dir", str(_REPO_ROOT / "downloads" / "lms")))
    out_dir.mkdir(parents=True, exist_ok=True)
    arpa_name = cfg.get("arpa_name", f"custom_{order}gram.arpa")
    arpa_path = out_dir / arpa_name
    klm_path = arpa_path.with_suffix(".klm")
    tmp_dir = out_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    text_arnoma = cfg.get("text_arpa_only", False)
    discount_fallback = cfg.get("discount_fallback", True)
    keep_arpa = bool(cfg.get("keep_arpa", True))

    corpus_path = _decompress_if_needed(Path(args.corpus), tmp_dir)

    # Build lmplz command.
    lmplz_cmd = [
        args.lmplz,
        "-o", str(order),
        "--text", str(corpus_path),
        "--arpa", str(arpa_path),
        "--prune", *(str(p) for p in prune),
    ]
    if discount_fallback:
        lmplz_cmd.append("--discount_fallback")

    logging.info(f"lmplz cmd: {' '.join(lmplz_cmd)}")
    if args.dry_run:
        logging.info("[dry-run] exiting.")
        return 0

    if shutil.which(args.lmplz) is None:
        logging.error(f"lmplz not found at '{args.lmplz}'. Install KenLM and add it to PATH.")
        return 1

    res = subprocess.run(lmplz_cmd, check=False)
    if res.returncode != 0:
        logging.error(f"lmplz failed with exit code {res.returncode}.")
        return res.returncode

    # Convert to binary KenLM for faster loading.
    if not text_arnoma and shutil.which(args.build_binary):
        bb_cmd = [args.build_binary, "-q", "8", "-b", "8", "trie", str(arpa_path), str(klm_path)]
        logging.info(f"build_binary cmd: {' '.join(bb_cmd)}")
        res = subprocess.run(bb_cmd, check=False)
        if res.returncode != 0:
            logging.warning(f"build_binary failed (rc={res.returncode}); ARPA kept at {arpa_path}.")
        else:
            logging.info(f"Wrote binary KenLM: {klm_path}")
            if not keep_arpa:
                arpa_path.unlink(missing_ok=True)
    else:
        logging.info(f"Skipping binary conversion (text_arpa_only={text_arnoma}, "
                     f"build_binary={shutil.which(args.build_binary)}).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
