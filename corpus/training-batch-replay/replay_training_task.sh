#!/bin/bash
# Extract replay config from a training task, then run replay_training_batch.sh.

set -eu -o pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 TASK_DIR" >&2
  exit 2
fi

TASK_DIR=$1
ENV_DIR=${ENV_DIR:-}
OUTPUT_DIR=${OUTPUT_DIR:-replayed_batches/$(basename "${TASK_DIR}")}
REPLAY_CONFIG=${REPLAY_CONFIG:-${OUTPUT_DIR}/replay_config.sh}
OUTPUT=${OUTPUT:-${OUTPUT_DIR}/global-batches.jsonl.gz}
DATA_CACHE_PATH=${DATA_CACHE_PATH:-${OUTPUT_DIR}/cache}
LOG_DIR=${LOG_DIR:-${OUTPUT_DIR}/logs}
OUTPUT_BASENAME=$(basename "${OUTPUT}")
LOG=${LOG:-${LOG_DIR}/${OUTPUT_BASENAME%.jsonl.gz}.log}

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG}") 2> >(tee -a "${LOG}" >&2)
echo "Logging to ${LOG}"

EXTRACT_ARGS=("${TASK_DIR}" --format sh --output "${REPLAY_CONFIG}")
if [[ -n "${ENV_DIR}" ]]; then
  EXTRACT_ARGS+=(--env-dir "${ENV_DIR}")
fi

python "${SCRIPT_DIR}/extract_replay_config.py" "${EXTRACT_ARGS[@]}"

if [[ -z "${ENV_DIR}" ]]; then
  # shellcheck disable=SC1090
  source "${REPLAY_CONFIG}"
fi

export ENV_DIR
export REPLAY_CONFIG
export OUTPUT
export DATA_CACHE_PATH

bash "${SCRIPT_DIR}/replay_training_batch.sh"
