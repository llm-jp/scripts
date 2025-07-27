#!/bin/bash
#
# llm-jp-eval v1.4.1 installation script
#
# This script use CPU on a cluster.
#  - In a SLURM environment, it is recommend to use CPU nodes.
#
# Usage:
# On a cluster with SLURM:
#   Run `sbatch --paratition {FIX_ME} install.sh TARGET_DIR`
# On a cluster without SLURM:
#   Run `bash install.sh TARGET_DIR > logs/install-eval.out 2> logs/install-eval.err`
# - TARGET_DIR: Instalation directory
#
#SBATCH --job-name=install-llm-jp-eval
#SBATCH --partition={FIX_ME}
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eux -o pipefail

if [ $# -ne 1 ]; then
  set +x
  >&2 echo Usage: sbatch \(or bash\)  install.sh TARGET_DIR
  exit 1
fi

INSTALLER_DIR=$(pwd)
TARGET_DIR=$1
INSTALLER_COMMON=$INSTALLER_DIR/../../../../common/installers.sh

>&2 echo INSTALLER_DIR=$INSTALLER_DIR
>&2 echo TARGET_DIR=$TARGET_DIR
>&2 echo INSTALLER_COMMON=$INSTALLER_COMMON
source $INSTALLER_COMMON

mkdir -p $TARGET_DIR
pushd $TARGET_DIR

# Copy basic scripts for llm-jp-eval
cp ${INSTALLER_DIR}/scripts/run_llm-jp-eval.sh .
mkdir resources
cp ${INSTALLER_DIR}/resources/config_*.yaml resources/
mkdir logs

ENV_DIR=${TARGET_DIR}/environment
mkdir $ENV_DIR
pushd $ENV_DIR

# Copy enviroment scripts
cp ${INSTALLER_DIR}/{install.sh,requirements*.txt} .
mkdir scripts
cp ${INSTALLER_DIR}/scripts/environment.sh scripts/
source scripts/environment.sh

# Record current environment variables
set > installer_envvar.log

# src is used to store all resources for from-scratch builds
mkdir src
pushd src

# Install Python (function in $INSTALLER_COMMON)
install_python v${PYTHON_VERSION} ${ENV_DIR}/python
popd # $ENV_DIR

# Prepare venv
python/bin/python3 -m venv venv-eval venv-vllm

# Install vllm
source venv-vllm/bin/activate
pip install --no-cache-dir -U pip setuptools wheel
# This implicitly installs vllm-flash-attn with their recommended version
pip install --no-cache-dir -r requirements-vllm.txt
deactivate

# Install llm-jp-eval
source venv-eval/bin/activate
python -m pip install --no-cache-dir -U pip setuptools wheel

pushd src
git clone https://github.com/llm-jp/jp-eval-customization.git -b v${LLM_JP_EVAL_TAG}
pushd llm-jp-eval
if [ -n "$LLM_JP_EVAL_BUG_FIX_COMMIT_IDS" ]; then
  git cherry-pick -m 1 ${LLM_JP_EVAL_BUG_FIX_COMMIT_IDS}
fi
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir -r ${ENV_DIR}/requirements-eval.txt
pip install --no-cache-dir -e .

INFERENCE_SCRIPT=offline_inference/vllm/offline_inference_vllm.py
# Remove execution time due to its complexity in handling
sed -i.bak 's/_{GENERATOR_TYPE}_{current_time}//' "$INFERENCE_SCRIPT"

# Fix the URL for JGLUE dataset download in multiple files
JGLUE_SCRIPTS=(
  "src/llm_jp_eval/jaster/jcommonsenseqa.py"
  "src/llm_jp_eval/jaster/jnli.py"
  "src/llm_jp_eval/jaster/jsquad.py"
  "src/llm_jp_eval/jaster/jsts.py"
)

for jglue_script in "${JGLUE_SCRIPTS[@]}"; do
  sed -i.bak 's|yahoojapan/JGLUE/main/datasets|yahoojapan/JGLUE/v1\.1\.0/datasets|g' "$jglue_script"
done

# Preprocess dataset
python scripts/preprocess_dataset.py \
  --dataset-name all-with-nc  \
  --output-dir ${ENV_DIR}/data/llm-jp-eval \
  --version-name $LLM_JP_EVAL_TAG

popd  #src
popd  # $ENV_DIR
popd  # $TARGET_DIR

# Check integrity of evaluation dataset
HASH_FILE=${INSTALLER_DIR}/resources/sha256sums.csv
DEV_DATASET_DIR=${ENV_DIR}/data/llm-jp-eval/${LLM_JP_EVAL_TAG}/evaluation/dev

set +x
declare -A hash_map

while IFS=$'\t' read -r filename hash; do
  hash_map["$filename"]="$hash"
done < "$HASH_FILE"

for file in ${DEV_DATASET_DIR}/*; do
  filename=$(basename "$file")
  calculated_hash=$(sha256sum "$file" | awk '{print $1}')

  if [[ "${hash_map[$filename]}" != "$calculated_hash" ]]; then
    >&2 echo "NG: $filename"
    >&2 echo "Expected: ${hash_map[$filename]}"
    >&2 echo "Got: $calculated_hash"
    echo "Aborted." | tee >(cat >&2)
    exit 1
  fi
done

echo "Installation done." | tee >(cat >&2)
