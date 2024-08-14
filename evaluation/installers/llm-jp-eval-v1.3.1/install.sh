#!/bin/bash
#
# llm-jp-eval v1.3.1 installation script
#
# This script consumes 1 node on the `cpu` partition on the cluster.
#
# Usage:
# On a cluster with SLURM:
#   Run `sbatch --paratition {partition} install.sh TARGET_DIR`
# On a cluster without SLURM:
#   Run `bash install.sh TARGET_DIR > logs/install-eval.out 2> logs/install-eval.err`
# - TARGET_DIR: Instalation directory
#
#SBATCH --job-name=install-llm-jp-eval
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eux

ENV_CHOICES=($(ls scripts/envs))
TARGET_ENV_MSG="Set TARGET_ENV from (${ENV_CHOICES} ) or add a new configuratione in 'scripts/envs'."

if [ $# -ne 2 ]; then
  set +x
  >&2 echo Usage: sbatch \(or bash\)  install.sh TARGET_ENV TARGET_DIR
  >&2 echo $TARGET_ENV_MSG
  exit 1
fi

INSTALLER_DIR=$(pwd)
TARGET_ENV=$1
TARGET_DIR=$2

if [[ ! " ${ENV_CHOICES[@]} " =~ " ${TARGET_ENV} " ]]; then
  set +x
  >&2 echo $TARGET_ENV_MSG
  exit 1
fi

>&2 echo INSTALLER_DIR=$INSTALLER_DIR
>&2 echo TARGET_DIR=$TARGET_DIR
>&2 echo TARGET_ENV=$TARGET_ENV

mkdir -p $TARGET_DIR
pushd $TARGET_DIR

# Copy basic scripts for llm-jp-eval
cp ${INSTALLER_DIR}/scripts/run_llm-jp-eval.sh .
mkdir resources
cp ${INSTALLER_DIR}/resources/config_base.yaml resources/
mkdir logs

ENV_DIR=${TARGET_DIR}/environment
mkdir $ENV_DIR
pushd $ENV_DIR

# Copy enviroment scripts
cp ${INSTALLER_DIR}/install.sh .
mkdir scripts

# Create environment.sh
BASE_ENV_SHELL=${INSTALLER_DIR}/scripts/env_common.sh
EXT_ENV_SHELL=${INSTALLER_DIR}/scripts/envs/${TARGET_ENV}/environment.sh
NEW_ENV_SHELL=scripts/environment.sh

touch $NEW_ENV_SHELL
echo -e "#!/bin/bash\n# from $BASE_ENV_SHELL" >> $NEW_ENV_SHELL
cat $BASE_ENV_SHELL >> $NEW_ENV_SHELL
echo -e "\n# from $EXT_ENV_SHELL" >> $NEW_ENV_SHELL
cat $EXT_ENV_SHELL >> $NEW_ENV_SHELL

source $NEW_ENV_SHELL

# Record current environment variables
set > installer_envvar.log

# src is used to store all resources for from-scratch builds
mkdir src
pushd src

# Install Python
git clone https://github.com/python/cpython -b v${PYTHON_VERSION}
pushd cpython
./configure --prefix="${ENV_DIR}/python" --enable-optimizations
make -j 64
make install
popd # src
popd # $ENV_DIR

# Prepare venv
python/bin/python3 -m venv venv
source venv/bin/activate
python -m pip install --no-cache-dir -U pip setuptools

# Install llm-jp-eval
pushd src
git clone https://github.com/llm-jp/llm-jp-eval -b v${LLM_JP_EVAL_TAG}
pushd llm-jp-eval
if [ -n "$LLM_JP_EVAL_BUG_FIX_COMMIT_IDS" ]; then
  git cherry-pick -m 1 ${LLM_JP_EVAL_BUG_FIX_COMMIT_IDS}
fi
pip install --no-cache-dir .

# Preprocess dataset
python scripts/preprocess_dataset.py \
  --dataset-name all  \
  --output-dir ${ENV_DIR}/data/llm-jp-eval \
  --version-name $LLM_JP_EVAL_TAG
popd  # src
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
  fi
done

echo "Installation done." | tee >(cat >&2)
