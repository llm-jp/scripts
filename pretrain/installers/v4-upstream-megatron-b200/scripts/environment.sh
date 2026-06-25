#!/bin/bash
# Environment variables and CUDA toolkit setup for the upstream Megatron-LM
# native (no-container) build on the B200 (Blackwell / sm_100) Slurm cluster.
#
# Port of pretrain/installers/v4-upstream-megatron-abci, adapted to:
#   * Slurm (cpu/gpu partitions) instead of ABCI PBS/qsub,
#   * CUDA 13 / torch cu130 / sm_100 instead of CUDA 12.6 / Hopper,
#   * FlashAttention-4 (Blackwell) instead of FlashAttention-3 (Hopper),
#   * prebuilt cu13 TransformerEngine wheel instead of a source build.
#
# This cluster does NOT module-load CUDA/cuDNN/NCCL for the build: we point
# CUDA_HOME at the system CUDA-13 toolkit and let cu13 wheels supply cuDNN/NCCL.

# ----- Megatron-LM source ----------------------------------------------------
export PRETRAIN_MEGATRON_REPO="${PRETRAIN_MEGATRON_REPO:-https://github.com/llm-jp/Megatron-LM}"
export PRETRAIN_MEGATRON_TAG="${PRETRAIN_MEGATRON_TAG:-core_v0.18.0_b200}"
# Ensure the appropriate Huggingface tokenizer is included.
export PRETRAIN_TOKENIZER_TAG="${PRETRAIN_TOKENIZER_TAG:-v3.0b2}"

# ----- Python ----------------------------------------------------------------
export PRETRAIN_PYTHON_VERSION=3.12   # managed by uv (no from-source CPython build)

# ----- CUDA 13 toolkit (matches torch cu130; provides nvcc + CCCL for APEX) ---
export PRETRAIN_CUDA_HOME="${PRETRAIN_CUDA_HOME:-/usr/local/cuda-toolkit/13.0.3}"
export PRETRAIN_CUDA_INDEX=cu130

# ----- toolchain versions ----------------------------------------------------
export PRETRAIN_TORCH_VERSION=2.12.0
export PRETRAIN_TRANSFORMER_ENGINE_VERSION=2.16.0
export PRETRAIN_CUBLAS_VERSION=13.5.1.27    # TE 2.16 needs cublasLtGroupedMatrixLayoutInit
export PRETRAIN_FLASH_ATTENTION_4_VERSION=4.0.0b17
export PRETRAIN_CUTLASS_DSL_VERSION=4.5.2   # quack 0.5.0 needs ThrMma (removed in 4.6.0.dev0)
export PRETRAIN_QUACK_VERSION=0.5.0
export PRETRAIN_APEX_COMMIT="${PRETRAIN_APEX_COMMIT:-}"   # empty = latest main (--depth 1)

# ----- Blackwell / B200 build flags ------------------------------------------
export TORCH_CUDA_ARCH_LIST=10.0
export NVTE_CUDA_ARCHS=100a     # a bare "100" leaves TE CMake's CUDA_ARCHITECTURES empty
export NVTE_FRAMEWORK=pytorch

# ----- CUDA toolkit on PATH (build nvcc + headers) ---------------------------
export CUDA_HOME="${PRETRAIN_CUDA_HOME}"
export CUDA_PATH="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${HOME}/.local/bin:${PATH}"
export CPATH="${CUDA_HOME}/targets/x86_64-linux/include/cccl:${CUDA_HOME}/include:${CPATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"

# ----- uv settings -----------------------------------------------------------
export UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

# Convenience: the venv python created by install_venv.sh.
export PY="${TARGET_DIR:-}/venv/bin/python"
