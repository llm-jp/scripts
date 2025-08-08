#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -N 0200_convert
#PBS -l select=1
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -m n

set -eu -o pipefail
cd "$PBS_O_WORKDIR"

JOBID=${PBS_JOBID%%.*}

# Experiment and env
EXP_DIR="/groups/gcg51557/experiments/0200_tokenizer_v4"
ENV_DIR="/groups/gcg51557/experiments/0208_ckpt_converter_cpu"
MEGATRON_PATH="${ENV_DIR}/src/Megatron-LM"
LOADER_SAVER_PATH="${ENV_DIR}/scripts/ckpt_converter"

# MCore torch_dist checkpoints root (contains iter_XXXXXXX subdirectories)
CKPT_ROOT=${TASK_DIR}/checkpoints

# Iteration number. Must be provided from env (-v ITER=...)
ITER=${ITER:?ITER env is required, e.g. 100000}
ITER_NAME=iter_$(printf %07d ${ITER})

# HuggingFace tokenizer and output directory
HF_TOKENIZER_PATH=${EXP_DIR}/src/tokenizer_hf/v4_alpha1.3a
OUTPUT_DIR=${TASK_DIR}/checkpoints_hf/${ITER_NAME}

# Save dtype: bfloat16 | float16 | float32
SAVE_DTYPE=bfloat16

echo "EXP_DIR=${EXP_DIR}"
echo "TASK_DIR=${TASK_DIR}"
echo "ITER=${ITER}"
echo "MEGATRON_PATH=${MEGATRON_PATH}"
echo "CKPT_ROOT=${CKPT_ROOT}"
echo "HF_TOKENIZER_PATH=${HF_TOKENIZER_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOADER_SAVER_PATH=${LOADER_SAVER_PATH}"

source ${ENV_DIR}/venv/bin/activate

# Force CPU
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# Logs
mkdir -p ${TASK_DIR}/logs
LOGFILE=${TASK_DIR}/logs/convert_${ITER}_${JOBID}.out
ERRFILE=${TASK_DIR}/logs/convert_${ITER}_${JOBID}.err
exec > "$LOGFILE" 2> "$ERRFILE"

# Sanity checks
test -f "${MEGATRON_PATH}/tools/checkpoint/convert.py"
test -f "${LOADER_SAVER_PATH}/loader_mcore_cpu.py" || { echo "Missing loader: ${LOADER_SAVER_PATH}/loader_mcore_cpu.py"; exit 1; }
test -f "${LOADER_SAVER_PATH}/saver_hf_llmjp.py" || { echo "Missing saver: ${LOADER_SAVER_PATH}/saver_hf_llmjp.py"; exit 1; }
test -d "${CKPT_ROOT}/${ITER_NAME}" || { echo "Missing checkpoint dir: ${CKPT_ROOT}/${ITER_NAME}"; exit 1; }
test -d "${HF_TOKENIZER_PATH}"      || { echo "Missing HF tokenizer path: ${HF_TOKENIZER_PATH}"; exit 1; }

# --- Prepare minimal torch_dist root for Megatron loader ---
TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT INT TERM

LOAD_ROOT="${TMP_DIR}/torch_dist_root"
mkdir -p "${LOAD_ROOT}" "${OUTPUT_DIR}"

echo "${ITER}" > "${LOAD_ROOT}/latest_checkpointed_iteration.txt"
ln -s "${CKPT_ROOT}/${ITER_NAME}" "${LOAD_ROOT}/${ITER_NAME}"

echo "Converting torch_dist -> HF on CPU..."

PYTHONPATH="${LOADER_SAVER_PATH}:${MEGATRON_PATH}:${PYTHONPATH:-}" \
python "${MEGATRON_PATH}/tools/checkpoint/convert.py" \
  --model-type GPT \
  --loader mcore_cpu \
  --loader-transformer-impl local \
  --position-embedding-type rope \
  --load-dir "${LOAD_ROOT}" \
  --saver hf_llmjp \
  --save-dir "${OUTPUT_DIR}" \
  --hf-tokenizer-path "${HF_TOKENIZER_PATH}" \
  --save-dtype "${SAVE_DTYPE}" \
  --no-checking \
  --megatron-path "${MEGATRON_PATH}"

echo "Done. Files created:"
find "${OUTPUT_DIR}" -type f | sort