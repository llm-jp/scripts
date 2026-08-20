#!/bin/bash
#
# Safety evaluation (LLM_Safety_Eva) installation script
#
# This script uses CPU on a cluster (a GPU is not required for installation).
#  - In a SLURM environment, it is recommended to use CPU nodes.
#
# Usage:
# On a cluster with SLURM:
#   Run `sbatch --partition {FIX_ME} install.sh TARGET_DIR [BENCHMARK_DATA_DIR]`
# On a cluster without SLURM:
#   Run `bash install.sh TARGET_DIR [BENCHMARK_DATA_DIR] > logs/install-safety-eval.out 2> logs/install-safety-eval.err`
# - TARGET_DIR: Installation directory
# - BENCHMARK_DATA_DIR: Directory holding the benchmark data JSON files
#   (default: ./LLM_Safety_Eva/benchmark_data next to this script).
#
# The benchmark data is NOT part of this repository (it is large and includes
# data derived from the gated AnswerCarefully dataset). Take the
# `LLM_Safety_Eva/benchmark_data/` directory from the distribution zip
# (LLM_Safety_Eva_Web-main.zip) and either place it next to this script at
# ./LLM_Safety_Eva/benchmark_data/ or pass its path as BENCHMARK_DATA_DIR.
#
# The JTruthfulQA classifier (nlp-waseda/roberta_jtruthfulqa) is prefetched
# into the environment-local HF cache so the evaluation phase never spends
# time downloading on compute nodes.
#
#SBATCH --job-name=install-safety-eval
#SBATCH --partition={FIX_ME}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eux -o pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  set +x
  >&2 echo Usage: sbatch \(or bash\) install.sh TARGET_DIR [BENCHMARK_DATA_DIR]
  exit 1
fi

INSTALLER_DIR=$(pwd)
TARGET_DIR=$(realpath $1)
BENCHMARK_DATA_DIR=${2:-${INSTALLER_DIR}/LLM_Safety_Eva/benchmark_data}

# The evaluation itself only needs the five benchmark files below; the
# distribution zip contains more (dev/mix splits etc.), which are copied too
# if present but not required.
REQUIRED_BENCHMARKS=(
  jbbq_age.json
  jtruthfulqa.json
  answer_carefully_test.json
  JSocialFact-01-test.json
  safety_boundary.json
)

if [ ! -d "${BENCHMARK_DATA_DIR}" ]; then
  set +x
  >&2 echo "ERROR: benchmark data directory not found: ${BENCHMARK_DATA_DIR}"
  >&2 echo "The benchmark data is distributed separately (LLM_Safety_Eva_Web-main.zip)."
  >&2 echo "Unzip it and place LLM_Safety_Eva/benchmark_data/ next to this script,"
  >&2 echo "or pass its path as the second argument."
  exit 1
fi
for f in "${REQUIRED_BENCHMARKS[@]}"; do
  if [ ! -f "${BENCHMARK_DATA_DIR}/${f}" ]; then
    >&2 echo "WARNING: ${BENCHMARK_DATA_DIR}/${f} not found; the corresponding benchmark will be unavailable at run time."
  fi
done

>&2 echo INSTALLER_DIR=$INSTALLER_DIR
>&2 echo TARGET_DIR=$TARGET_DIR
>&2 echo BENCHMARK_DATA_DIR=$BENCHMARK_DATA_DIR

mkdir -p $TARGET_DIR
pushd $TARGET_DIR

# Copy basic scripts for the safety evaluation
cp ${INSTALLER_DIR}/scripts/run_safety-eval.sh .
mkdir -p logs

# Copy the evaluation code as received from the safety WG (vendored under
# safety-eval/LLM_Safety_Eva in the installer repository; as-is except for
# the OpenAI-compatible judge endpoint support in
# evaluators/llm_as_a_judge_chatgpt.py).
mkdir -p LLM_Safety_Eva
cp ${INSTALLER_DIR}/LLM_Safety_Eva/*.py LLM_Safety_Eva/
cp ${INSTALLER_DIR}/LLM_Safety_Eva/config.yaml LLM_Safety_Eva/
cp ${INSTALLER_DIR}/LLM_Safety_Eva/README.md LLM_Safety_Eva/
cp ${INSTALLER_DIR}/LLM_Safety_Eva/llm_safety_latest_*.txt LLM_Safety_Eva/
for d in eva_prompt evaluators model utils; do
  mkdir -p LLM_Safety_Eva/$d
  cp ${INSTALLER_DIR}/LLM_Safety_Eva/$d/* LLM_Safety_Eva/$d/
done

# Stage the benchmark data (see the header note; not in the repository)
mkdir -p LLM_Safety_Eva/benchmark_data
cp ${BENCHMARK_DATA_DIR}/*.json LLM_Safety_Eva/benchmark_data/ 2>/dev/null || true
cp ${BENCHMARK_DATA_DIR}/*.csv LLM_Safety_Eva/benchmark_data/ 2>/dev/null || true

ENV_DIR=${TARGET_DIR}/environment
mkdir -p $ENV_DIR
pushd $ENV_DIR

# Copy environment scripts
cp ${INSTALLER_DIR}/install.sh .
cp ${INSTALLER_DIR}/requirements.txt .

# Use uv from PATH, or install a standalone uv under the environment when
# unavailable (same pattern as the llm-jp-judge installer).
export UV_PYTHON_INSTALL_DIR="${ENV_DIR}/python"
if ! command -v uv >/dev/null 2>&1; then
  export UV_INSTALL_DIR="${ENV_DIR}/uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${UV_INSTALL_DIR}:${PATH}"
fi

# Record current environment variables
set > installer_envvar.log

# Python 3.10 matches the environment the safety WG verified
# (LLM_Safety_Eva/llm_safety_latest_working.txt).
uv venv venv --python 3.10
uv pip install --python venv/bin/python -r requirements.txt

# Prefetch the JTruthfulQA classifier into the environment-local HF cache;
# run_safety-eval.sh points HF_HOME here (with HF_HUB_OFFLINE=1) during the
# JTruthfulQA evaluation phase.
mkdir -p ${ENV_DIR}/data/hf
HF_HOME=${ENV_DIR}/data/hf venv/bin/python - <<'PYEOF'
from huggingface_hub import snapshot_download

snapshot_download("nlp-waseda/roberta_jtruthfulqa")
print("prefetched: nlp-waseda/roberta_jtruthfulqa")
PYEOF

popd  # $ENV_DIR
popd  # $TARGET_DIR

echo "Installation done." | tee >(cat >&2)
