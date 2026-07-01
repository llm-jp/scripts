#!/bin/bash
#SBATCH --job-name=swift-convert
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/convert-%j.out
#SBATCH --error=logs/convert-%j.err

set -eu -o pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)}"
cd "${SCRIPT_DIR}"
mkdir -p logs

: "${VENV_DIR:?VENV_DIR is not set}"
: "${MEGATRON_CKPT_STEP:?MEGATRON_CKPT_STEP is not set}"
: "${MCORE_PATH:?MCORE_PATH is not set}"
: "${HF_REF_WEIGHT:?HF_REF_WEIGHT is not set}"
: "${HF_OUTPUT_DIR:?HF_OUTPUT_DIR is not set}"
: "${MODELSCOPE_CACHE:?MODELSCOPE_CACHE is not set}"

VENV_DIR=$(realpath -eP "${VENV_DIR}")
PY="${VENV_DIR}/venv/bin/python"
if [ ! -x "${PY}" ]; then
    >&2 echo "Missing python: ${PY}"
    exit 1
fi
export PATH="${VENV_DIR}/venv/bin:${PATH}"

PADDED=$(printf "%07d" "${MEGATRON_CKPT_STEP}")
ITER_NAME="iter_${PADDED}"
SRC_ITER_DIR="${MCORE_PATH}/${ITER_NAME}"
if [ ! -d "${SRC_ITER_DIR}" ]; then
    >&2 echo "Missing checkpoint dir: ${SRC_ITER_DIR}"
    exit 1
fi
if [ ! -d "${HF_REF_WEIGHT}" ]; then
    >&2 echo "Missing HF_REF_WEIGHT dir: ${HF_REF_WEIGHT}"
    exit 1
fi

TMP_DIR=$(mktemp -d)
cleanup() {
    rm -rf "${TMP_DIR}"
}
trap cleanup EXIT INT TERM

MCORE_ROOT="${TMP_DIR}/mcore"
mkdir -p "${MCORE_ROOT}" "${HF_OUTPUT_DIR}"
echo "${MEGATRON_CKPT_STEP}" > "${MCORE_ROOT}/latest_checkpointed_iteration.txt"
ln -s "$(realpath -eP "${SRC_ITER_DIR}")" "${MCORE_ROOT}/${ITER_NAME}"

OUT_DIR="${HF_OUTPUT_DIR}/${ITER_NAME}"
MODEL_TYPE="${MODEL_TYPE:-qwen3_moe}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
TEMPLATE="${TEMPLATE:-qwen3}"

echo "VENV_DIR=${VENV_DIR}"
echo "MCORE_ROOT=${MCORE_ROOT}"
echo "MEGATRON_CKPT_STEP=${MEGATRON_CKPT_STEP}"
echo "SRC_ITER_DIR=$(realpath -eP "${SRC_ITER_DIR}")"
echo "MCORE_ROOT_LATEST=$(cat "${MCORE_ROOT}/latest_checkpointed_iteration.txt")"
echo "MCORE_ROOT_ITER_LINK=$(readlink -f "${MCORE_ROOT}/${ITER_NAME}")"
echo "HF_REF_WEIGHT=${HF_REF_WEIGHT}"
echo "OUT_DIR=${OUT_DIR}"
echo "MODEL_TYPE=${MODEL_TYPE}"
echo "TEMPLATE=${TEMPLATE}"

swift export \
  --model "${HF_REF_WEIGHT}" \
  --model_type "${MODEL_TYPE}" \
  --template "${TEMPLATE}" \
  --mcore_model "${MCORE_ROOT}" \
  --to_hf true \
  --torch_dtype "${TORCH_DTYPE}" \
  --output_dir "${OUT_DIR}" \
  ${EXTRA_SWIFT_EXPORT_ARGS:-}

echo "Done. Files created:"
find "${OUT_DIR}" -type f | sort
