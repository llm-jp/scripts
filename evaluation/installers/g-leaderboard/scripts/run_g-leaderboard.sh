#!/bin/bash
#SBATCH --job-name=g-leaderboard
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
TOKENIZER=$MODEL
WANDB_ENTITY=llm-jp-eval
WANDB_PROJECT=test

# Fixed vars
G_LEADERBOARD_DIR=${ENV_DIR}/src/g-leaderboard
CONFIG_DIR=${G_LEADERBOARD_DIR}/configs
SCRIPT_PATH=${G_LEADERBOARD_DIR}/scripts/run_eval.py

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

# Blended run config settings
BLENDED_RUN_CONFIG=resources/blended_run_config.yaml
BLENDED_RUN_CONFIG_DIR=${G_LEADERBOARD_DIR}/blend_run_configs
cp $BLENDED_RUN_CONFIG ${BLENDED_RUN_CONFIG_DIR}/config.yaml

# Create a temporal project
TMP_G_LEADERBOARD_DIR=$(mktemp -d "${HOME}/ckpt_convert.XXXXXXXX")
cp -r $G_LEADERBOARD_DIR/* $TMP_G_LEADERBOARD_DIR
cp $NEW_CONFIG $TMP_G_LEADERBOARD_DIR/configs/config.yaml

# Run g-leaderboard
pushd $TMP_G_LEADERBOARD_DIR
python $SCRIPT_PATH

# Clean up
popd
rm -rf $TMP_G_LEADERBOARD_DIR

echo "Done"
