#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -N xxx_conv
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:mpiprocs=8:ngpus=8
#PBS -l walltime=168:00:00
#PBS -j oe
#PBS -koed
#PBS -V
#PBS -o outputs/convert/
#PBS -m n

set -eu -o pipefail

# Setup environment
source /etc/profile.d/modules.sh
module load cuda/12.1/12.1.1
module load cudnn/9.5/9.5.1
module load hpcx/2.20
module load nccl/2.23/2.23.4-1

source venv/bin/activate

JOB_ID=$(echo $PBS_JOBID | cut -d. -f1)
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE | hostname -f)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

NUM_NODES=$(sort -u $PBS_NODEFILE | wc -l)
NUM_GPUS_PER_NODE=8

## Debug/logging flags
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_LOGDEST_DBG=stderr
export CUDNN_LOGERR_DBG=1

NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))

TOKENIZER_MODEL="llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model"
MEGATRON_PATH="/path/to/default/megatron"
# Path configurations with defaults
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH:-}

WORLD_SIZE=$((${TARGET_TP_SIZE} * ${TARGET_EP_SIZE} * ${TARGET_PP_SIZE}))

echo "Starting checkpoint conversion process..."
echo "Processing iterations from $START_ITER to $END_ITER with step size $STEP_SIZE"

# Tokenizer configuration with default
TOKENIZER_DIR=src/llm-jp-tokenizer/hf/ver3.0/llm-jp-tokenizer-100k.ver3.0b2/

# イテレーションの配列を生成
declare -a ITERATIONS=()

# STEP_SIZEごとにイテレーションを生成
for ((iter = ${START_ITER}; iter < ${END_ITER}; iter += ${STEP_SIZE})); do
    ITERATIONS+=($iter)
done

# 最後のイテレーションが配列になければ追加
if [[ ! " ${ITERATIONS[@]} " =~ " ${END_ITER} " ]]; then
    ITERATIONS+=($END_ITER)
fi

echo "Processing the following iterations: ${ITERATIONS[@]}"

for iter in "${ITERATIONS[@]}"; do
    # Format iteration number with leading zeros (7 digits)
    ITER_FORMATTED=$(printf "iter_%07d" $iter)

    # Construct full paths
    LOAD_CHECKPOINT_PATH="${BASE_LOAD_PATH}/${ITER_FORMATTED}"
    SAVE_CHECKPOINT_PATH="${BASE_SAVE_PATH}/${ITER_FORMATTED}"

    echo "========================================"
    echo "Processing iteration ${iter}..."
    echo "Loading from: ${LOAD_CHECKPOINT_PATH}"
    echo "Saving to: ${SAVE_CHECKPOINT_PATH}"

    # Check if the checkpoint directory exists
    if [ ! -d "${LOAD_CHECKPOINT_PATH}" ]; then
        echo "Warning: Source directory ${LOAD_CHECKPOINT_PATH} does not exist. Skipping."
        continue
    fi

    # Create save directory
    mkdir -p "${SAVE_CHECKPOINT_PATH}"

    # Run conversion script
    python scripts/sakura/ckpt/mcore_to_hf_mixtral.py \
        --load_path "${LOAD_CHECKPOINT_PATH}" \
        --save_path "${SAVE_CHECKPOINT_PATH}" \
        --target_tensor_model_parallel_size ${TARGET_TP_SIZE} \
        --target_pipeline_model_parallel_size ${TARGET_PP_SIZE} \
        --target_expert_model_parallel_size ${TARGET_EP_SIZE} \
        --target_params_dtype "bf16" \
        --world_size ${WORLD_SIZE} \
        --convert_checkpoint_from_megatron_to_transformers \
        --print-checkpoint-structure

    # Check if conversion was successful
    if [ $? -eq 0 ]; then
        # Copy tokenizer files
        cp -r ${TOKENIZER_DIR}/* "${SAVE_CHECKPOINT_PATH}"
        echo "Successfully processed iteration ${iter}"
    else
        echo "Error processing iteration ${iter}"
        echo "Skipping tokenizer copy for this iteration"
    fi

    echo "----------------------------------------"
done

echo "Checkpoint conversion process completed"
