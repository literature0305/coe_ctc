#!/usr/bin/env bash
# =============================================================================
# coe-ctc — Evaluation Runner
# =============================================================================
# Decodes a CTC checkpoint on LibriSpeech (or libri-light) splits and reports
# WER / CER / latency per split. Supports:
#   * checkpoint averaging (top-N best by validation WER),
#   * N-gram LM rescoring (pyctcdecode + KenLM),
#   * greedy or beam decoding,
#   * remote-server job submission via the ssub shim.
#
# =============================================================================
# Usage Examples
# =============================================================================
#
# [1] Greedy decoding on all 4 LibriSpeech splits
#   bash scripts/evaluation/run_evaluation.sh \
#       --models outputs/ctc_tiny_300bpe/best_checkpoints \
#       --config scripts/evaluation/configs/libri.yaml
#
# [2] Checkpoint averaging (top-10) + A100 × 1 job
#   bash scripts/evaluation/run_evaluation.sh \
#       --models outputs/ctc_small_3000bpe/best_checkpoints \
#       --config scripts/evaluation/configs/libri_light.yaml \
#       --checkpoint-avg 10 --mode job --ngpu 1 --gpu-type A100
#
# [3] 3-gram pruned LM rescoring (openslr-11)
#   bash scripts/evaluation/run_evaluation.sh \
#       --models outputs/ctc_small_3000bpe/best_checkpoints \
#       --config scripts/evaluation/configs/libri.yaml \
#       --ngram downloads/lms/3-gram.pruned.1e-7.arpa.gz
#
# [4] 4-gram (full) LM rescoring
#   bash scripts/evaluation/run_evaluation.sh \
#       --models outputs/ctc_small_3000bpe/best_checkpoints \
#       --config scripts/evaluation/configs/libri.yaml \
#       --ngram downloads/lms/4-gram.arpa.gz
#
# [4b] 4-gram + LibriSpeech lexicon (closed-vocabulary fusion — best WER)
#      Requires `--with-lm` in preprocess to fetch librispeech-vocab.txt:
#   bash scripts/evaluation/run_evaluation.sh \
#       --models outputs/ctc_small_3000bpe/best_checkpoints \
#       --config scripts/evaluation/configs/libri.yaml \
#       --ngram downloads/lms/4-gram.arpa.gz \
#       --lexicon downloads/lms/librispeech-vocab.txt \
#       --beam 16 --alpha 0.5 --beta 1.5
#
# [5] Beam search (default greedy)
#   bash scripts/evaluation/run_evaluation.sh \
#       --models outputs/ctc_small_3000bpe/best_checkpoints \
#       --config scripts/evaluation/configs/libri.yaml \
#       --beam 4
#
# [6] Decode a single .pt file (not a directory)
#   bash scripts/evaluation/run_evaluation.sh \
#       --models outputs/ctc_small_3000bpe/best_checkpoints/step=0050000-wer=0.0512.pt \
#       --config scripts/evaluation/configs/libri.yaml
#
# [7] Smoke test (truncate each split to 5 batches)
#   bash scripts/evaluation/run_evaluation.sh \
#       --models outputs/ctc_small_3000bpe/best_checkpoints \
#       --config scripts/evaluation/configs/libri.yaml \
#       --max-batches 5
#
# [8] CPU-only evaluation
#   bash scripts/evaluation/run_evaluation.sh \
#       --models outputs/ctc_small_3000bpe/best_checkpoints \
#       --config scripts/evaluation/configs/libri.yaml \
#       --device cpu
#
# =============================================================================
# Options
# =============================================================================
#   --models PATH                       (required) best_checkpoints/ dir OR a .pt file
#   --config PATH                       (required) eval YAML
#   --mode local|job                    default: local
#   --gpu-type A100|H100|2080ti         default: A100
#   --ngpu N                            default: 1
#   --num-workers N                     default: 8
#   --beam N                            default: 1 (greedy)
#   --ngram PATH                        KenLM ARPA / binary
#   --lexicon PATH                      Word-list (librispeech-vocab.txt / lexicon.txt)
#                                         used as pyctcdecode `unigrams=`. Closed-vocab fusion.
#   --alpha F                           LM weight (default 0.5)
#   --beta F                            word-bonus (default 1.5)
#   --checkpoint-avg N                  average top-N best by valid WER (default 1)
#   --data ALIAS                        override eval YAML's data_alias
#   --output-dir DIR                    override default outputs/<run>/evaluation/
#   --device cpu|cuda|cuda:N            default: auto
#   --amp auto|bfloat16|float16|off     default: auto
#   --max-batches N                     truncate each split for smoke tests
#   --dry-run                           print plan / command, do not run
#   --priority N                        ssub priority (default 3)
#   --exp-id N                          ssub exp-id (default 370)
#   --job-name NAME                     ssub job name (default: eval-{stem})
#
# All unrecognized args are forwarded to coe_ctc.decoding.evaluate.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVAL_MODULE="coe_ctc.decoding.evaluate"
SSUB_PY="${REPO_ROOT}/src/coe_ctc/utils/ssub.py"

MODE="local"
MODELS=""
CONFIG=""
GPU_TYPE="A100"
NGPU="1"
NUM_WORKERS="8"
JOB_NAME=""
DRY_RUN=false
PRIORITY="3"
EXP_ID="370"
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            sed -n '2,98p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        --mode)         MODE="$2"; shift 2 ;;
        --gpu-type)     GPU_TYPE="$2"; shift 2 ;;
        --ngpu)         NGPU="$2"; shift 2 ;;
        --num-workers)  NUM_WORKERS="$2"; PASSTHROUGH+=("--num-workers" "$2"); shift 2 ;;
        --job-name)     JOB_NAME="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; PASSTHROUGH+=("--dry-run"); shift ;;
        --priority)     PRIORITY="$2"; shift 2 ;;
        --exp-id)       EXP_ID="$2"; shift 2 ;;
        --models)       MODELS="$2"; PASSTHROUGH+=("--models" "$2"); shift 2 ;;
        --config)       CONFIG="$2"; PASSTHROUGH+=("--config" "$2"); shift 2 ;;
        *)              PASSTHROUGH+=("$1"); shift ;;
    esac
done

if [[ -z "${MODELS}" ]]; then
    echo "Error: --models is required." >&2
    exit 1
fi
if [[ -z "${CONFIG}" ]]; then
    echo "Error: --config is required." >&2
    exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NUM_WORKERS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${NUM_WORKERS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${NUM_WORKERS}}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-${NUM_WORKERS}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if [[ "${MODE}" == "job" ]]; then
    JOB_NAME="${JOB_NAME:-eval-$(basename "${CONFIG}" .yaml)}"
    echo "======================================================================"
    echo "  coe-ctc evaluation job submission"
    echo "  models       : ${MODELS}"
    echo "  config       : ${CONFIG}"
    echo "  gpu-type     : ${GPU_TYPE}"
    echo "  ngpu         : ${NGPU}"
    echo "  job-name     : ${JOB_NAME}"
    echo "  dry-run      : ${DRY_RUN}"
    echo "======================================================================"

    SUBMIT_CMD=(python "${SSUB_PY}"
        --config "${CONFIG}"
        --gpu-type "${GPU_TYPE}"
        --ngpu "${NGPU}"
        --priority "${PRIORITY}"
        --exp-id "${EXP_ID}"
        --job-name "${JOB_NAME}"
        --train-module "${EVAL_MODULE}"
        --extra-args ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
    )
    ${DRY_RUN} && SUBMIT_CMD+=(--dry-run)
    echo "Submission cmd: ${SUBMIT_CMD[*]}"
    "${SUBMIT_CMD[@]}"
    exit 0
fi

echo "============================================================"
echo "coe-ctc Evaluation"
echo "  models   : ${MODELS}"
echo "  config   : ${CONFIG}"
echo "  args     : ${PASSTHROUGH[*]}"
echo "============================================================"

python3 -m "${EVAL_MODULE}" ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
