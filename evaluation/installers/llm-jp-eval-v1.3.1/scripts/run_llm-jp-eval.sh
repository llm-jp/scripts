#!/bin/bash
#SBATCH --job-name=llm-jp-eval
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=200G
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
TOKENIZER=$MODEL
WANDB_ENTITY=llm-jp-eval
WANDB_PROJECT=test

# Fixed vars
CONFIG_DIR=${ENV_DIR}/src/llm-jp-eval/configs
SCRIPT_PATH=${ENV_DIR}/src/llm-jp-eval/scripts/evaluate_llm.py
DATASET_DIR=${ENV_DIR}/data/llm-jp-eval/${LLM_JP_EVAL_TAG}/evaluation/dev

# Config settings
NEW_CONFIG=${CONFIG_DIR}/config.${WANDB_PROJECT}.${WANDB_RUN_NAME}.yaml
REPLACE_VARS=("MODEL" "TOKENIZER" "DATASET_DIR" "WANDB_ENTITY" "WANDB_PROJECT" "WANDB_RUN_NAME")

# Create a new config file to save the config file of each run
cp $CONFIG_TEMPLATE $NEW_CONFIG

# Replace variables
for VAR in "${REPLACE_VARS[@]}"; do
  VALUE=$(eval echo \${$VAR})
  sed -i "s|<<${VAR}>>|${VALUE}|g" $NEW_CONFIG
done

# Run llm-jp-eval
python $SCRIPT_PATH -cn $(basename $NEW_CONFIG)

echo "Done"
