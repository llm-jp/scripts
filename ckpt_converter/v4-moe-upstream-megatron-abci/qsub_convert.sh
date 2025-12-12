#!/bin/bash
#PBS -P gch51639
#PBS -q R9920251000
#PBS -N convert
#PBS -v RTYPE=rt_HG
#PBS -l select=1:ngpus=1
#PBS -l walltime=168:00:00
#PBS -j oe
#PBS -koed
#PBS -V
#PBS -o outputs/
#PBS -m n

set -eu -o pipefail

cd "$PBS_O_WORKDIR"
mkdir -p outputs

: "${VENV_DIR:?VENV_DIR is not set}"
: "${MEGATRON_CKPT_STEP:?MEGATRON_CKPT_STEP is not set}"
: "${MCORE_PATH:?MCORE_PATH is not set}"
: "${HF_REF_WEIGHT:?HF_REF_WEIGHT is not set}"
: "${HF_OUTPUT_DIR:?HF_OUTPUT_DIR is not set}"
: "${MODELSCOPE_CACHE:?MODELSCOPE_CACHE is not set}"
: "${MEGATRON_LM_PATH:?MEGATRON_LM_PATH is not set}"

source "${VENV_DIR}/bin/activate"
source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
module load cudnn/9.5/9.5.1
module load nccl/2.25/2.25.1-1
module load hpcx/2.20

export MEGATRON_CKPT_STEP
export MODELSCOPE_CACHE
export MEGATRON_LM_PATH

PADDED=$(printf "%07d" "$MEGATRON_CKPT_STEP")
JOB_ID=$(echo "$PBS_JOBID" | cut -d. -f1)
export MASTER_ADDR=$(head -n 1 "$PBS_NODEFILE" | xargs hostname -f)
export MASTER_PORT=$((10000 + (JOB_ID % 50000)))
export CUDA_VISIBLE_DEVICES=0

HF_OUTPUT_DIR="${HF_OUTPUT_DIR}/iter_${PADDED}"

swift export \
    --model "${HF_REF_WEIGHT}" \
    --model_type qwen3_moe \
    --mcore_model "${MCORE_PATH}" \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir "${HF_OUTPUT_DIR}"
