#!/bin/bash
#SBATCH --job-name=llm-jp-eval
#SBATCH --partition=<partition>
#SBATCH --exclusive
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -e

# module load
source scripts/environment.sh

# open file limit
ulimit -n 65536 1048576

# python virtualenv
source venv/bin/activate

# script path
SCRIPT_PATH=src/llm-jp-eval/scripts/evaluate_llm.py

# run llm-jp-eval
python $SCRIPT_PATH \
  -cn config.yaml \
  model.pretrained_model_name_or_path=$1 \
  tokenizer.pretrained_model_name_or_path=$1 \
  dataset_dir=dataset/llm-jp-eval/${LLM_JP_EVAL_TAG}/evaluation/dev \
  wandb.project=$2 \
  wandb.run_name=$3 \

echo "Done"
