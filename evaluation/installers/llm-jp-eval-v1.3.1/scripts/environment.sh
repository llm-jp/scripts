#!/bin/bash
# List of environment variables and module loads for llm-jp-eval


export PYTHON_VERSION=3.10.14

export LLM_JP_EVAL_TAG=1.3.1
export LLM_JP_EVAL_BUG_FIX_COMMIT_IDS="9716bcc"

export HOSTNAME=$(hostname)
case "$HOSTNAME" in
  "login2")
    # sakura
    export CUDA_VERSION_MAJOR=12
    export CUDA_VERSION_MINOR=1
    export CUDA_VERSION=${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}
    export CUDNN_VERSION=8.9.4
    module load cuda/${CUDA_VERSION}
    module load /data/cudnn-tmp-install/modulefiles/${CUDNN_VERSION}
    ;;
  *)
    # llm-jp, llm-jp-nvlink
    ;;
esac
