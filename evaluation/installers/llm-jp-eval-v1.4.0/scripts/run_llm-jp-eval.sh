#!/bin/bash
#SBATCH --job-name=llm-jp-eval
#SBATCH --partition=<partition>
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eux

# Open file limit
ulimit -n 65536 1048576

ENV_DIR=environment
source ${ENV_DIR}/scripts/environment.sh
source ${ENV_DIR}/venv/bin/activate

# Arguments
MODEL=$1
WANDB_RUN_NAME=$2

# Semi-fixed vars
CONFIG_TEMPLATE=resources/config_base.yaml
OFFLINE_CONFIG_TEMPLATE=resources/config_offline_inference_vllm.yaml
TOKENIZER=$MODEL
WANDB_ENTITY=llm-jp-eval
WANDB_PROJECT=test

# Fixed vars
EVAL_DIR=${ENV_DIR}/src/llm-jp-eval
CONFIG_DIR=${EVAL_DIR}/configs
OFFLINE_SCRIPT_PATH=${EVAL_DIR}/offline_inference/vllm/offline_inference_vllm.py
DATASET_DIR=${ENV_DIR}/data/llm-jp-eval/${LLM_JP_EVAL_TAG}/evaluation/dev
OFFLINE_OUT=vllm-outputs/${WANDB_ENTITY}/${WANDB_PROJECT}

# Config settings
MODEL_NORM=$(echo "$MODEL" | sed 's/\//--/g')
NEW_CONFIG=${CONFIG_DIR}/config.${WANDB_PROJECT}.${WANDB_RUN_NAME}.yaml
NEW_OFFLINE_CONFIG=${OFFLINE_OUT}/${MODEL_NORM}/config_offline_inference_vllm.yaml

REPLACE_VARS=("MODEL" "TOKENIZER" "DATASET_DIR" "WANDB_ENTITY" "WANDB_PROJECT" "WANDB_RUN_NAME" "OFFLINE_OUT" "MODEL_NORM")

# Create a new config file to save the config file of each run
cp $CONFIG_TEMPLATE $NEW_CONFIG
mkdir -p $(dirname $NEW_OFFLINE_CONFIG)
cp $OFFLINE_CONFIG_TEMPLATE $NEW_OFFLINE_CONFIG

# Replace variables
for VAR in "${REPLACE_VARS[@]}"; do
  VALUE=$(eval echo \${$VAR})
  sed -i "s|<<${VAR}>>|${VALUE}|g" $NEW_CONFIG
  sed -i "s|<<${VAR}>>|${VALUE}|g" $NEW_OFFLINE_CONFIG
done

# Offline generation
# Need to generate dump for each run because $OFFLINE_SCRIPT_PATH read this dump
python ${EVAL_DIR}/scripts/dump_prompts.py -cn $(basename $NEW_CONFIG)
# Generate outputs and copy data of $NEW_CONFIG to output via $NEW_OFFLINE_CONFIG
python $OFFLINE_SCRIPT_PATH -cp $(pwd)/$(dirname $NEW_OFFLINE_CONFIG)

# Run llm-jp-eval
python ${EVAL_DIR}/scripts/evaluate_llm.py -cn $(basename $NEW_CONFIG)

echo "Done"
