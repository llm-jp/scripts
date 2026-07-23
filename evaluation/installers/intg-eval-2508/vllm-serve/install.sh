#!/bin/bash
#
# vllm-serve mode installer: copies the serve-mode scripts into an existing
# intg-eval experiment environment and applies a small compatibility patch to
# the swallow harness's (otherwise unused) openai_completions.py.
#
# Prerequisites: the target environments (swallow_v202411[-tf5],
# llm-jp-eval-v2.1.x) are already installed under TARGET_DIR.
#
# Usage:
#   bash install.sh TARGET_DIR
#   - TARGET_DIR: the environment/ directory of the experiment
#                 (e.g. /data/experiments/<exp>/environment)

set -eux -o pipefail

if [ $# -ne 1 ]; then
  set +x
  >&2 echo "Usage: bash install.sh TARGET_DIR (the environment/ directory)"
  exit 1
fi

INSTALLER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARGET_DIR=$(realpath "$1")

SERVE_DIR=${TARGET_DIR}/vllm-serve
mkdir -p "$SERVE_DIR"
cp "${INSTALLER_DIR}"/scripts/serve_common.sh \
   "${INSTALLER_DIR}"/scripts/run_eval_serve.sh \
   "${INSTALLER_DIR}"/scripts/run-swallow-serve.sh \
   "${INSTALLER_DIR}"/scripts/run_llm-jp-eval-serve.sh \
   "${INSTALLER_DIR}"/scripts/inference_openai.py \
   "${INSTALLER_DIR}"/scripts/compare_results.py \
   "$SERVE_DIR/"

# Apply the local-completions compatibility patch to each installed swallow
# environment. openai_completions.py is not used by the offline (run-eval.sh)
# flow, so this does not change existing behavior.
PATCH_FILE=${INSTALLER_DIR}/patches/openai_completions-serve-compat.patch
PATCH_TARGET=lm-evaluation-harness-en/lm_eval/models/openai_completions.py
for swallow_env in "${TARGET_DIR}"/swallow_v202411*/; do
  repo=${swallow_env}environment/src/swallow-evaluation
  [ -d "$repo" ] || continue
  pushd "$repo"
  if git apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
    echo "Patch already applied in ${repo}; skipping."
  elif git apply --check "$PATCH_FILE" 2>/dev/null; then
    git apply "$PATCH_FILE"
  else
    # An older revision of this patch is likely applied; the file is not
    # touched by the offline flow, so restore it and apply the current one.
    git checkout -- "$PATCH_TARGET"
    git apply "$PATCH_FILE"
  fi
  popd
done

echo "Installation done." | tee >(cat >&2)
