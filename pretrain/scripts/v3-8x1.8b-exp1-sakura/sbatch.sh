#!/bin/bash
#SBATCH --job-name=0059_v3-8x1.8b-exp1
#SBATCH --partition=gpu-small
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

set -eu -o pipefail

source scripts/environment.sh
source venv/bin/activate

# CUTLASS
CUTLASS_HOME=src/cutlass/build
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUTLASS_HOME}/lib

export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS_PER_NODE=$(echo $SLURM_TASKS_PER_NODE | cut -d '(' -f 1)
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))

echo NUM_NODES=$NUM_NODES
echo NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE
echo NUM_GPUS=$NUM_GPUS

MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=1024
GRADIENTS_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE / NUM_GPUS))

if [ $GRADIENTS_ACCUMULATION_STEPS -lt 1 ]; then
  echo "Global batch size is too small for the number of GPUs"
  exit 1
fi

WEIGHT_DECAY=0.1
GRAD_CLIP=1

# deepspeed config
DEEPSPEED_CONFIG="src/moe-recipes/deepspeed_config.json"

BF16_ENABLED=true
DEEPSPEED_ZERO_STAGE=3

OVERLAP_COMMUNICATION=true
CONTINOUS_GRADIENTS=true

DEEPSPEED_SUB_GROUP_SIZE=1e12
DEEPSPEED_REDUCE_BUCKET_SIZE=1e9
DEEPSPEED_STAGE3_PREFETCH_BUCKET_SIZE=5e8
DEEPSPEED_STAGE3_PARAM_PERSISTENCE_THRESHOLD=1e6

DEEPSPEED_STAGE3_MAX_LIVE_PARAMETERS=1e9
DEEPSPEED_STAGE3_MAX_REUSE_DISTANCE=1e9

WALL_CLOCK_BREAKDOWN=false

DEEPSPEED_CONGIG_CONTENT=$(
  cat <<EOF
{
  "bf16": {
    "enabled": $BF16_ENABLED
  },
  "data_types": {
    "grad_accum_dtype": "fp32"
  },
  "zero_optimization": {
    "stage": $DEEPSPEED_ZERO_STAGE,
    "overlap_comm": $OVERLAP_COMMUNICATION,
    "contiguous_gradients": $CONTINOUS_GRADIENTS,
    "sub_group_size": $DEEPSPEED_SUB_GROUP_SIZE,
    "reduce_bucket_size": $DEEPSPEED_REDUCE_BUCKET_SIZE,
    "stage3_prefetch_bucket_size": $DEEPSPEED_STAGE3_PREFETCH_BUCKET_SIZE,
    "stage3_param_persistence_threshold": $DEEPSPEED_STAGE3_PARAM_PERSISTENCE_THRESHOLD,
    "stage3_max_live_parameters": $DEEPSPEED_STAGE3_MAX_LIVE_PARAMETERS,
    "stage3_max_reuse_distance": $DEEPSPEED_STAGE3_MAX_REUSE_DISTANCE
  },
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "gradient_accumulation_steps": $GRADIENTS_ACCUMULATION_STEPS,
  "gradient_clipping": $GRAD_CLIP,
  "wall_clock_breakdown": $WALL_CLOCK_BREAKDOWN
}
EOF
)

# write deepspeed config file
echo "$DEEPSPEED_CONGIG_CONTENT" >$DEEPSPEED_CONFIG

# Initialization
python example/checkpoint_init.py

PYTHONPATH=${PYTHONPATH:-}

if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="./src/moe-recipes:./src/moe-recipes/src"
else
    export PYTHONPATH="./src/moe-recipes:./src/moe-recipes/src:${PYTHONPATH}"
fi

echo "PYTHONPATH is now: $PYTHONPATH"

mpirun \
  -np $NUM_GPUS \
  --npernode $NUM_GPUS_PER_NODE \
  -bind-to none \
  -map-by slot \
  -x TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -x LD_LIBRARY_PATH \
  -x PATH \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x NUM_NODES=$NUM_NODES \
  -x NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE \
  -x GRADIENTS_ACCUMULATION_STEPS=$GRADIENTS_ACCUMULATION_STEPS \
  -x MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE \
  -x GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE \
  -x DEEPSPEED_CONFIG=$DEEPSPEED_CONFIG \
  -x DEEPSPEED_ZERO_STAGE=$DEEPSPEED_ZERO_STAGE \
  -x PYTHONPATH=$PYTHONPATH \
  bash scripts/v3-8x1.8b-exp1-sakura/train.sh
