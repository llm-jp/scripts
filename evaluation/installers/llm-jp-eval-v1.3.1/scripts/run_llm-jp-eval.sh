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

# open file limit
ulimit -n 65536 1048576

ENV_DIR=environment
source ${ENV_DIR}/scripts/environment.sh
source ${ENV_DIR}/venv/bin/activate

# dynamic vars
MODEL=$1
WANDB_RUN_NAME=$2

# semi-fixed vars
CONFIG_TEMPLATE=resources/config_base.yaml
TOKENIZER=$MODEL
WANDB_ENTITY=llm-jp-eval
WANDB_PROJECT=test

# fixed vars
CONFIG_DIR=${ENV_DIR}/src/llm-jp-eval/configs
SCRIPT_PATH=${ENV_DIR}/src/llm-jp-eval/scripts/evaluate_llm.py
DATASET_DIR=${ENV_DIR}/data/llm-jp-eval/${LLM_JP_EVAL_TAG}/evaluation/dev

# config settings
NEW_CONFIG=${CONFIG_DIR}/config.${WANDB_PROJECT}.${WANDB_RUN_NAME}.yaml
REPLACE_VARS=("MODEL" "TOKENIZER" "DATASET_DIR" "WANDB_ENTITY" "WANDB_PROJECT" "WANDB_RUN_NAME")

# create config
cp $CONFIG_TEMPLATE $NEW_CONFIG

# replace variables
for VAR in "${REPLACE_VARS[@]}"; do
  VALUE=$(eval echo \${$VAR})
  sed -i "s|<<${VAR}>>|${VALUE}|g" $NEW_CONFIG
done

# run llm-jp-eval
python $SCRIPT_PATH -cn $(basename $NEW_CONFIG)

echo "Done"
