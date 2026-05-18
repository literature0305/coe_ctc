#!/usr/bin/env bash
# =============================================================================
# coe-ctc — KenLM N-gram trainer
# =============================================================================
# Trains a KenLM N-gram LM from a custom corpus and (optionally) compiles it
# to a binary KenLM ``.klm`` for fast loading at decode time.
#
# Prerequisites:
#   * KenLM compiled with the `lmplz` and `build_binary` binaries on PATH.
#       sudo apt-get install -y libboost-all-dev cmake
#       git clone https://github.com/kpu/kenlm.git && cd kenlm
#       mkdir build && cd build && cmake .. && make -j
#       export PATH=$PWD/bin:$PATH
#
# =============================================================================
# Usage Examples
# =============================================================================
#
# [1] Train a 4-gram from the default LibriSpeech normalized corpus
#   bash scripts/training/train_ngram.sh \
#       --config scripts/training/configs/4gram.yaml \
#       --corpus downloads/corpus/librispeech-lm-norm.txt.gz
#
# [2] Train from a plain-text corpus (no .gz)
#   bash scripts/training/train_ngram.sh \
#       --config scripts/training/configs/4gram.yaml \
#       --corpus path/to/my_corpus.txt
#
# [3] Override order / output dir via --set
#   bash scripts/training/train_ngram.sh \
#       --config scripts/training/configs/4gram.yaml \
#       --corpus downloads/corpus/librispeech-lm-norm.txt.gz \
#       --set order=3 --set arpa_name=custom_3gram.arpa
#
# [4] Dry run — print the lmplz command but don't execute
#   bash scripts/training/train_ngram.sh \
#       --config scripts/training/configs/4gram.yaml \
#       --corpus downloads/corpus/librispeech-lm-norm.txt.gz \
#       --dry-run
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG=""
CORPUS=""
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            sed -n '2,46p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        --config) CONFIG="$2"; PASSTHROUGH+=("--config" "$2"); shift 2 ;;
        --corpus) CORPUS="$2"; PASSTHROUGH+=("--corpus" "$2"); shift 2 ;;
        *) PASSTHROUGH+=("$1"); shift ;;
    esac
done

if [[ -z "${CONFIG}" || -z "${CORPUS}" ]]; then
    echo "Error: both --config and --corpus are required." >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "============================================================"
echo "coe-ctc — KenLM N-gram training"
echo "  config   : ${CONFIG}"
echo "  corpus   : ${CORPUS}"
echo "  passthr  : ${PASSTHROUGH[*]}"
echo "============================================================"

python3 -m coe_ctc.lm.train_ngram ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
