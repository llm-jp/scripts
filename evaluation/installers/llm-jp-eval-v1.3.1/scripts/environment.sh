#!/bin/bash
# List of environment variables and module loads for llm-jp-eval

export CUDA_VERSION_MAJOR=12
export CUDA_VERSION_MINOR=1
export CUDA_VERSION=${PRETRAIN_CUDA_VERSION_MAJOR}.${PRETRAIN_CUDA_VERSION_MINOR}
export CUDNN_VERSION=8.9.4

export PYTHON_VERSION=3.10.14

export LLM_JP_EVAL_TAG=v1.3.1
export LLM_JP_EVAL_BUG_FIX_COMMIT_IDS=9716bcc

HOSTNAME=$(hostname)
case "$HOSTNAME" in
    "login2")
        # sakura
        module load cuda/${PRETRAIN_CUDA_VERSION}
        module load /data/cudnn-tmp-install/modulefiles/${PRETRAIN_CUDNN_VERSION}
        ;;
    *)
        # llm-jp, llm-jp-nvlink
        ;;
esac
