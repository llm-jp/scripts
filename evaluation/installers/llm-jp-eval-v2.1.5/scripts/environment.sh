#!/bin/bash
export PYTHON_VERSION=3.11

export LLM_JP_EVAL_TAG=2.1.5
# Commit pointed to by the v2.1.5 tag
export LLM_JP_EVAL_COMMIT_HASH=5067fe7bcd33797643835573505d5ec06858ea34

# llm-jp-eval-inference has no release tags; latest commit as of 2026-07-20
# (Merge PR #16: handle_vllm_update; vllm>=0.19.1 / torch>=2.10.0)
export LLM_JP_EVAL_INFERENCE_COMMIT_HASH=c6cd0fa8ce1e891d9904dfeb49a4e532349166a6

export LLM_JP_EVAL_BUG_FIX_COMMIT_IDS=
