#!/usr/bin/env bash
# =============================================================================
# coe-ctc — Training Launcher
# =============================================================================
# Unified entry for single-GPU debug, multi-GPU torchrun, and remote-server
# `--mode job` submission. Mirrors the pattern in
#   tsm-trainer008_aed/.../scripts/forecasting/training/train.sh
#
# =============================================================================
# Usage Examples
# =============================================================================
#
# ── Local Training ────────────────────────────────────────────────────────
#
# [1] Single GPU (2080 Ti, debug)
#   bash scripts/training/train.sh \
#       --data libri100 \
#       --config scripts/training/configs/transformer_120m_2080ti.yaml
#
# [2] Multi-GPU single node — torchrun auto-detected from torch.cuda.device_count()
#   bash scripts/training/train.sh \
#       --data libri960 \
#       --config scripts/training/configs/conformer_120m_a100x4.yaml
#
# [3] Multi-node (mpirun) — auto-detected from hostfile $HOSTFILE
#   bash scripts/training/train.sh \
#       --data libri960 \
#       --config scripts/training/configs/conformer_300m_a100x8.yaml
#
# [4] Resume from checkpoint
#   bash scripts/training/train.sh \
#       --data libri960 \
#       --config scripts/training/configs/conformer_300m_a100x8.yaml \
#       --resume-from-checkpoint outputs/ctc_conformer_300m_3000bpe/checkpoints/latest.pt
#
# [5] Override hyperparameters from the CLI
#   bash scripts/training/train.sh \
#       --data libri100 \
#       --config scripts/training/configs/conformer_30m_2080ti.yaml \
#       --max-steps 50 --num-workers 4
#
# [5b] Use a different BPE vocab than the config's default (e.g. reuse the
#      libri960_bpe3000 model with a config that expected libri100_bpe300):
#   bash scripts/training/train.sh \
#       --data libri100 \
#       --config scripts/training/configs/transformer_30m_2080ti.yaml \
#       --n_bpe 3000
#   # → resolves data.bpe_model to downloads/bpe/libri100_bpe3000/spm.model
#
# ── Job Submission (Space/ssub) ───────────────────────────────────────────
#
# [6] Job — A100 × 8 (single node)
#   bash scripts/training/train.sh --mode job \
#       --data libri960 --ngpu 8 --gpu-type A100 \
#       --config scripts/training/configs/conformer_300m_a100x8.yaml
#
# [7] Job — A100 × 4
#   bash scripts/training/train.sh --mode job \
#       --data libri960 --ngpu 4 --gpu-type A100 \
#       --config scripts/training/configs/conformer_120m_a100x4.yaml
#
# [8] Job — multi-node A100 × 24 (3 nodes × 8 GPUs)
#   bash scripts/training/train.sh --mode job \
#       --data libri960 --ngpu 24 --gpu-type A100 \
#       --config scripts/training/configs/conformer_800m_a100x8.yaml
#
# [9] Dry-run (print the ssub command, do not submit)
#   bash scripts/training/train.sh --mode job \
#       --data libri960 --ngpu 8 --gpu-type A100 \
#       --config scripts/training/configs/conformer_300m_a100x8.yaml \
#       --dry-run
#
# [10] Custom job name + override output dir
#   bash scripts/training/train.sh --mode job \
#       --data libri960 --ngpu 8 --gpu-type A100 \
#       --config scripts/training/configs/conformer_300m_a100x8.yaml \
#       --job-name conf-300m-exp1 \
#       --output-dir /group-volume/.../outputs/exp1
#
# =============================================================================
# Options
# =============================================================================
#   --data libri100|libri960|libri_light           (required)
#   --config PATH                                  (required) training YAML
#   --mode local|job                               default: local
#   --gpu-type A100|H100|2080ti                    default: A100 (job mode)
#   --ngpu N                                       default: 8    (job mode)
#   --num-workers N                                default: 8    cap for OMP/MKL
#   --job-name NAME                                default: train-{stem}
#   --output-dir DIR                               override config output_dir
#   --resume-from-checkpoint PATH
#   --max-steps N
#   --seed N
#   --dry-run                                      print plan/command, do not run
#   --priority N                                   ssub priority (default 3)
#   --exp-id N                                     ssub experiment id (default 370)
#
# All other args are passed through to python -m coe_ctc.training.train.
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_MODULE="coe_ctc.training.train"
JOB_SUBMIT_PY="${REPO_ROOT}/src/coe_ctc/utils/ssub.py"   # produced in Phase 4
HOSTFILE="${HOSTFILE:-/horovod/generated/hostfile}"
MASTER_PORT="${MASTER_PORT:-29500}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# ─────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────
MODE="local"
DATA=""
CONFIG=""
GPU_TYPE="A100"
NGPU="8"
NUM_WORKERS="8"
JOB_NAME=""
OUTPUT_DIR=""
RESUME=""
DRY_RUN=false
PRIORITY="3"
EXP_ID="370"
PASSTHROUGH=()

# ─────────────────────────────────────────────────────────────────────────
# Parse
# ─────────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            sed -n '2,80p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        --mode)        MODE="$2"; shift 2 ;;
        --data)        DATA="$2"; PASSTHROUGH+=("--data" "$2"); shift 2 ;;
        --config)      CONFIG="$2"; PASSTHROUGH+=("--config" "$2"); shift 2 ;;
        --gpu-type)    GPU_TYPE="$2"; shift 2 ;;
        --ngpu)        NGPU="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; PASSTHROUGH+=("--num-workers" "$2"); shift 2 ;;
        --job-name)    JOB_NAME="$2"; shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; PASSTHROUGH+=("--output-dir" "$2"); shift 2 ;;
        --resume-from-checkpoint) RESUME="$2"; PASSTHROUGH+=("--resume-from-checkpoint" "$2"); shift 2 ;;
        --dry-run)     DRY_RUN=true; PASSTHROUGH+=("--dry-run"); shift ;;
        --priority)    PRIORITY="$2"; shift 2 ;;
        --exp-id)      EXP_ID="$2"; shift 2 ;;
        *)             PASSTHROUGH+=("$1"); shift ;;
    esac
done

if [[ -z "${DATA}" ]]; then
    echo "Error: --data is required (libri100 | libri960 | libri_light)." >&2
    exit 1
fi
if [[ -z "${CONFIG}" ]]; then
    echo "Error: --config is required." >&2
    exit 1
fi
if [[ "${MODE}" != "local" && "${MODE}" != "job" ]]; then
    echo "Error: --mode must be 'local' or 'job' (got '${MODE}')." >&2
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────
# Thread caps — same pattern as tsm-trainer008_aed/.../train.sh
# ─────────────────────────────────────────────────────────────────────────
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NUM_WORKERS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${NUM_WORKERS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${NUM_WORKERS}}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-${NUM_WORKERS}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# ─────────────────────────────────────────────────────────────────────────
# MODE: JOB  →  delegate to ssub shim (Phase 4 produces utils/ssub.py;
# we fall back to printing the command if the shim is missing).
# ─────────────────────────────────────────────────────────────────────────
if [[ "${MODE}" == "job" ]]; then
    JOB_NAME="${JOB_NAME:-train-$(basename "${CONFIG}" .yaml)}"
    echo "======================================================================"
    echo "  coe-ctc job submission"
    echo "  config       : ${CONFIG}"
    echo "  data         : ${DATA}"
    echo "  gpu-type     : ${GPU_TYPE}"
    echo "  ngpu         : ${NGPU}"
    echo "  job-name     : ${JOB_NAME}"
    echo "  output-dir   : ${OUTPUT_DIR:-<from config>}"
    echo "  resume       : ${RESUME:-<none>}"
    echo "  priority     : ${PRIORITY}"
    echo "  exp-id       : ${EXP_ID}"
    echo "  dry-run      : ${DRY_RUN}"
    echo "======================================================================"

    if [[ -x "${JOB_SUBMIT_PY}" ]] || [[ -f "${JOB_SUBMIT_PY}" ]]; then
        SUBMIT_CMD=(python "${JOB_SUBMIT_PY}"
            --config "${CONFIG}"
            --data "${DATA}"
            --gpu-type "${GPU_TYPE}"
            --ngpu "${NGPU}"
            --priority "${PRIORITY}"
            --exp-id "${EXP_ID}"
            --job-name "${JOB_NAME}"
            --train-module "${TRAIN_MODULE}"
        )
        [[ -n "${OUTPUT_DIR}" ]] && SUBMIT_CMD+=(--output-dir "${OUTPUT_DIR}")
        [[ -n "${RESUME}"     ]] && SUBMIT_CMD+=(--resume-from-checkpoint "${RESUME}")
        ${DRY_RUN} && SUBMIT_CMD+=(--dry-run)
        echo "Submission cmd: ${SUBMIT_CMD[*]}"
        "${SUBMIT_CMD[@]}"
    else
        echo "Warning: ${JOB_SUBMIT_PY} not found (Phase 4 produces it). Printing"
        echo "         the local-equivalent command instead. Run it on the remote node:"
        echo
        echo "  python -m ${TRAIN_MODULE} ${PASSTHROUGH[*]}"
    fi
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────
# MODE: LOCAL
# ─────────────────────────────────────────────────────────────────────────
if [[ -f "${HOSTFILE}" ]]; then
    NUM_NODES=$(wc -l < "${HOSTFILE}")
    GPUS_PER_NODE=$(head -1 "${HOSTFILE}" | grep -oP 'slots=\K[0-9]+' || echo "1")
    TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
    FIRST_HOST=$(head -1 "${HOSTFILE}" | awk '{print $1}')
    export MASTER_ADDR="${MASTER_ADDR:-$(python3 -c "import socket; print(socket.gethostbyname('${FIRST_HOST}'))" 2>/dev/null || echo "${FIRST_HOST}")}"
    export MASTER_PORT

    echo "============================================================"
    echo "coe-ctc Multi-Node Training (mpirun)"
    echo "  hostfile      : ${HOSTFILE}"
    echo "  num_nodes     : ${NUM_NODES}"
    echo "  gpus_per_node : ${GPUS_PER_NODE}"
    echo "  total_gpus    : ${TOTAL_GPUS}"
    echo "  master        : ${MASTER_ADDR}:${MASTER_PORT}"
    echo "  data          : ${DATA}"
    echo "  config        : ${CONFIG}"
    echo "============================================================"

    mpirun --allow-run-as-root \
        -np "${TOTAL_GPUS}" -bind-to none -map-by slot \
        --hostfile "${HOSTFILE}" -mca pml ob1 -mca btl ^openib \
        -x MASTER_ADDR -x MASTER_PORT -x NCCL_DEBUG="${NCCL_DEBUG}" \
        -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_IB_DISABLE=1 \
        -x OMP_NUM_THREADS -x MKL_NUM_THREADS -x OPENBLAS_NUM_THREADS \
        -x NUMEXPR_MAX_THREADS -x TOKENIZERS_PARALLELISM \
        -x PATH -x PYTHONPATH -x LD_LIBRARY_PATH \
        -x http_proxy -x https_proxy -x no_proxy \
        -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
        python3 -m "${TRAIN_MODULE}" ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
    exit $?
fi

# No hostfile → single-node fallback. Detect GPUs.
GPUS_PER_NODE=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")

echo "============================================================"
echo "coe-ctc Single-Node Training"
echo "  gpus          : ${GPUS_PER_NODE}"
echo "  data          : ${DATA}"
echo "  config        : ${CONFIG}"
echo "  num_workers   : ${NUM_WORKERS}"
echo "  args          : ${PASSTHROUGH[*]}"
echo "============================================================"

if [[ "${GPUS_PER_NODE}" -gt 1 ]]; then
    torchrun --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}" \
        -m "${TRAIN_MODULE}" ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
else
    python3 -m "${TRAIN_MODULE}" ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
fi
