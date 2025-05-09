#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -N 0156_convert
#PBS -l select=1
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -m n

cd $PBS_O_WORKDIR

JOBID=${PBS_JOBID%%.*}
mkdir -p ${TASK_DIR}/logs
LOGFILE=${TASK_DIR}/logs/convert-$JOBID.out
ERRFILE=${TASK_DIR}/logs/convert-$JOBID.err
exec > $LOGFILE 2> $ERRFILE

set -eu -o pipefail

# Arguments
EXPERIMENT_DIR=/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction
SCRIPT_DIR=${EXPERIMENT_DIR}/scripts/pretrain/scripts/v4-midtraining/midtrain
ENV_DIR=${EXPERIMENT_DIR}/environments
echo "EXPERIMENT_DIR=${EXPERIMENT_DIR}"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "TASK_DIR=${TASK_DIR}"
echo "PARAM_NAME=${PARAM_NAME}"
echo "ITER=${ITER}"

# Setup environment
source ${SCRIPT_DIR}/common/setup.sh

export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE | hostname -f)
export MASTER_PORT=$((10000 + RANDOM % 1000))
echo "hostname: ${MASTER_ADDR}"

ITER_NAME=iter_$(printf %07d ${ITER})  # iter_0123456

MEGATRON_PATH=${ENV_DIR}/src/Megatron-LM
TOKENIZER_MODEL_PATH=${ENV_DIR}/src/llm-jp-tokenizer/hf/ver3.0/llm-jp-tokenizer-100k.ver3.0b2
OUTPUT_DIR=${TASK_DIR}/checkpoints_hf/${ITER_NAME}

# Setup working directory
TEMP_DIR=$(mktemp -d "${HOME}/converter_${JOBID}_XXXXXX")
echo "TEMP_DIR=${TEMP_DIR}"
function rm_tempdir {
    if [ -e ${TEMP_DIR} ]; then
        echo "Removing remporary directory: ${TEMP_DIR}"
        rm -rf ${TEMP_DIR}
        echo "Done removing"
    fi
}
trap rm_tempdir EXIT
trap 'trap - EXIT; rm_tempdir; exit 1' INT PIPE TERM

########
# Step 1: Convert `torch_dist` format to `torch`
# This process requires to launch the trainer script with the same parallelism configs.
########
echo "Start converting: torch_dist --> torch"

# Prepare source model at specific iteration
mkdir ${TEMP_DIR}/torch_dist
echo ${ITER} > ${TEMP_DIR}/torch_dist/latest_checkpointed_iteration.txt
ln -s ${TASK_DIR}/checkpoints/${ITER_NAME} ${TEMP_DIR}/torch_dist/${ITER_NAME}

# Load ALL_PARAMS
source ${SCRIPT_DIR}/params/${PARAM_NAME}.sh
# Remove wandb params
EXCLUDE_KEYS=("--wandb-entity" "--wandb-project" "--wandb-exp-name")
NEW_PARAMS=()
skip_next=0
for param in "${ALL_PARAMS[@]}"; do
    if [[ $skip_next -eq 1 ]]; then
        skip_next=0
        continue
    fi
    for key in "${EXCLUDE_KEYS[@]}"; do
        if [[ "$param" == "$key" ]]; then
            skip_next=1
            continue 2
        fi
    done
    NEW_PARAMS+=("$param")
done
ALL_PARAMS=("${NEW_PARAMS[@]}")

# Add params specific to model conversion
ALL_PARAMS+=(
    --load ${TEMP_DIR}/torch_dist
    --ckpt-convert-format torch
    --ckpt-convert-save ${TEMP_DIR}
)
echo "ALL_PARAMS: ${ALL_PARAMS[@]}"

NUM_NODES=$(wc -l < $PBS_NODEFILE)
NUM_GPUS_PER_NODE=8
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))
echo "nnodes: ${NUM_NODES}; ngpus: ${NUM_GPUS}"
echo NUM_NODES=$NUM_NODES
echo NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE
echo NUM_GPUS=$NUM_GPUS

export NVTE_FUSED_ATTN=0
# Launch trainer script to convert the checkpoint
mpirun \
    --display-allocation \
    --report-bindings \
    --oversubscribe \
    -np ${NUM_GPUS} \
    --npernode ${NUM_GPUS_PER_NODE} \
    -bind-to none \
    -map-by slot \
    python ${MEGATRON_PATH}/pretrain_gpt.py \
        ${ALL_PARAMS[@]}

#echo "Files created by the Step 1:"
find ${TEMP_DIR}/torch | sort

########
# Step 2: Convert `torch` to `Hugging Face Llama2`
########

echo "Start converting: torch --> hf"

python ${MEGATRON_PATH}/tools/checkpoint/convert.py \
    --model-type GPT \
    --loader mcore \
    --saver llmjp4_hf \
    --load-dir ${TEMP_DIR}/torch \
    --save-dir ${OUTPUT_DIR} \
    --hf-tokenizer-path ${TOKENIZER_MODEL_PATH} \
    --save-dtype bfloat16 \
    --loader-transformer-impl transformer_engine \
    --megatron-path ${MEGATRON_PATH}

echo "Files created by the Step 2:"
find ${OUTPUT_DIR} | sort

########
# Step 3: Replace tokenizer model
########

echo "Start replacing tokenizer"

cp ${TOKENIZER_MODEL_PATH}/* ${OUTPUT_DIR}

echo "Final model files:"
find ${OUTPUT_DIR} | sort

echo "Done processing"
