#!/bin/bash
#SBATCH --job-name=0095_hf_to_nemo
#SBATCH --partition=FIXME
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -eu -o pipefail

if [ $# -ne 4 ]; then
  >&2 echo Usage: $0 ENV_DIR HPARAMS INPUT_HF OUTPUT_NEMO
  >&2 echo Example: $0 /path/to/env /path/to/param.yaml /path/to/hf/iter_XXX /path/to/nemo/iter_XXX
  exit 1
fi

ENV_DIR=$1; shift
HPARAMS_FILE=$1; shift
INPUT_DIR=$1; shift
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

# run
python scripts/pretrain/scripts/v3-cpt-instruct/tuning/hf_to_nemo.py \
  --input-name-or-path ${INPUT_DIR} \
  --output-path ${OUTPUT_DIR} \
  --hparams-file ${HPARAMS_FILE} \
  --cpu-only \
  --n-jobs 96

mkdir -p ${OUTPUT_DIR}
tar -xvf ${OUTPUT_DIR}.nemo -C ${OUTPUT_DIR}
rm ${OUTPUT_DIR}.nemo

echo Done
