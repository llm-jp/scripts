#!/bin/bash
#SBATCH --job-name=0095_nemo_to_hf
#SBATCH --partition=FIXME
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -eu -o pipefail

if [ $# -ne 4 ]; then
  >&2 echo Usage: $0 ENV_DIR INPUT_NEMO INPUT_HF OUTPUT_NEMO
  >&2 echo Example: $0 foo /path/to/env /path/to/nemo /path/to/hf/iter_XXX /path/to/tuned/iter_XXX
  exit 1
fi

ENV_DIR=$1; shift
INPUT_NEMO_DIR=$1; shift
INPUT_HF_DIR=$1; shift
OUTPUT_DIR=$1; shift

# module load
export MODULEPATH=/data/modules:${MODULEPATH}
module load cuda-12.1.1
module load cudnn-8.9.7
module load hpcx-2.17.1
module load nccl-2.18.3

# open file limit
ulimit -n 65536 1048576

source ${ENV_DIR}/venv/bin/activate

# create symlink to tuned parameters
weight_rel=$(ls ${INPUT_NEMO_DIR} | egrep '^step=[0-9]+?' | head -1)
echo "Weight rel: ${weight_rel}"
ln -s "${weight_rel}" ${INPUT_NEMO_DIR}/model_weights

# run
python scripts/pretrain/scripts/v3-cpt-instruct/tuning/nemo_to_hf.py \
  --input-name-or-path ${INPUT_NEMO_DIR} \
  --input-hf-path ${INPUT_HF_DIR} \
  --output-path ${OUTPUT_DIR} \
  --cpu-only \
  --n-jobs 96

echo Done
