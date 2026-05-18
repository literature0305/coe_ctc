#!/usr/bin/env bash
# =============================================================================
# coe-ctc — Preprocess Runner
# =============================================================================
# Downloads LibriSpeech / libri-light, builds Lhotse manifests, extracts
# fBank-80 features, and trains one or more BPE SentencePiece models (in
# parallel).
#
# All heavy lifting is delegated to scripts/preprocess/_download_libri.py
# (which lives in this same directory). This wrapper:
#   * caps CPU thread libraries to --num-workers (default 8),
#   * propagates --skip-* / --dry-run flags,
#   * ensures the repo's `src/` is on PYTHONPATH.
#
# =============================================================================
# Usage Examples
# =============================================================================
#
# [1] Standard 960h prep + both BPE 3000 and 300 in parallel
#     (100h subset manifests are derived automatically):
#   bash scripts/preprocess/run_preprocess.sh \
#       --data libri960 --n_bpe 3000 300
#
# [2] 100h only (smaller subset, single BPE = 500):
#   bash scripts/preprocess/run_preprocess.sh \
#       --data libri100 --n_bpe 500
#
# [3] libri-light unlabeled audio (small split, single BPE = 3000 reused from libri960):
#   bash scripts/preprocess/run_preprocess.sh \
#       --data libri_light --n_bpe 3000
#
# [4] Re-run with a CPU budget (cluster-friendly):
#   bash scripts/preprocess/run_preprocess.sh \
#       --data libri960 --n_bpe 3000 --num-workers 4
#
# [5] Re-extract fBank only (download/manifest/BPE skipped):
#   bash scripts/preprocess/run_preprocess.sh \
#       --data libri960 --n_bpe 3000 \
#       --skip-download --skip-manifest --skip-bpe
#
# [6] Train an extra BPE vocab against an existing prep:
#   bash scripts/preprocess/run_preprocess.sh \
#       --data libri960 --n_bpe 5000 \
#       --skip-download --skip-manifest --skip-fbank
#
# [7] libri-light medium split (~5500h):
#   bash scripts/preprocess/run_preprocess.sh \
#       --data libri_light --libri-light-splits medium --num-workers 16
#
# [8] Dry-run — print plan, do nothing:
#   bash scripts/preprocess/run_preprocess.sh --data libri960 --n_bpe 3000 --dry-run
#
# [9] Also download LM bundle (vocab + 3-gram pruned + 4-gram from openslr-11)
#     for N-gram fusion decoding:
#   bash scripts/preprocess/run_preprocess.sh \
#       --data libri960 --n_bpe 3000 --with-lm
#
# [10] Download every LM resource (lexicon, full 4G corpus, all ARPAs):
#   bash scripts/preprocess/run_preprocess.sh \
#       --data libri960 --n_bpe 3000 --with-lm all
#
# [11] Download a specific subset (e.g. just the lexicon for k2/icefall decode):
#   bash scripts/preprocess/run_preprocess.sh \
#       --data libri960 --n_bpe 3000 --with-lm lexicon vocab
#
# =============================================================================
# Options
# =============================================================================
#   --data {libri100|libri960|libri_light}      (required)
#   --n_bpe N [N ...]                           BPE vocab sizes (default: 3000)
#   --download-dir DIR                          Root for downloads/ (default: ./downloads)
#   --num-workers N                             Max parallel CPU workers (default: 8)
#   --num-mel-bins N                            fBank dim (default: 80)
#   --sampling-rate N                           Audio sampling rate (default: 16000)
#   --bpe-model-type {unigram|bpe|char}         (default: unigram)
#   --libri-light-splits S [S ...]              Override libri-light splits (default: small)
#   --skip-download                             Skip raw-audio download/extract
#   --skip-manifest                             Skip Lhotse manifest building
#   --skip-fbank                                Skip fBank feature extraction
#   --skip-bpe                                  Skip SentencePiece training
#   --with-lm [RESOURCE ...]                    Also download LM files from openslr-11.
#                                                 No args = default bundle (vocab + 3-gram pruned + 4-gram).
#                                                 "all"   = lexicon + corpus + every ARPA.
#                                                 Any subset of: vocab lexicon corpus 3gram 3gram-pruned 4gram
#   --dry-run                                   Print plan and exit
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/_download_libri.py"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
NUM_WORKERS="8"
PASSTHROUGH_ARGS=()
WANT_DATA=""

# Parse so we can intercept --num-workers for thread-cap exports.
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            sed -n '2,86p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            PASSTHROUGH_ARGS+=("--num-workers" "$2")
            shift 2
            ;;
        --data)
            WANT_DATA="$2"
            PASSTHROUGH_ARGS+=("--data" "$2")
            shift 2
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${WANT_DATA}" ]]; then
    echo "Error: --data is required (libri100 | libri960 | libri_light)." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# CPU thread caps — same pattern as tsm-trainer008_aed/.../train.sh.
# Without these, NumPy / MKL / OpenBLAS / PyArrow spawn os.cpu_count() threads
# per process, which can crash shared servers.
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NUM_WORKERS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${NUM_WORKERS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${NUM_WORKERS}}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-${NUM_WORKERS}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "======================================================================"
echo "  coe-ctc preprocess"
echo "  data         : ${WANT_DATA}"
echo "  num_workers  : ${NUM_WORKERS}"
echo "  OMP/MKL/OBLAS: ${NUM_WORKERS}"
echo "  PYTHONPATH   : ${PYTHONPATH}"
echo "  script       : ${PY_SCRIPT}"
echo "  arguments    : ${PASSTHROUGH_ARGS[*]}"
echo "======================================================================"

exec python3 "${PY_SCRIPT}" "${PASSTHROUGH_ARGS[@]}"
