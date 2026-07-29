"""Prefetch Hub-hosted dependencies of the swallow EN suite at install time.

Run inside venv-harness with HF_DATASETS_CACHE / HF_MODULES_CACHE pointed at
the environment-local cache (see install.sh). run-eval.sh switches to that
cache and enables HF_DATASETS_OFFLINE / HF_EVALUATE_OFFLINE when it exists,
so evaluation jobs do not depend on Hub access.
"""

import os
import shutil
import sys
import traceback

from pathlib import Path

import evaluate


# Metric modules resolved from the Hub at run time: exact_match is loaded at
# lm_eval.api.metrics import time, squad_v2 by the squadv2 task.
for module in ("exact_match", "squad_v2"):
    evaluate.load(module)
    print(f"prefetched evaluate module: {module}", flush=True)

# Resolution can additionally cache *comparison* variants of the same module
# names. lm_eval only ever wants the metric variants, and a comparison copy
# in the cache is exactly the poisoning that broke process_results on ABCI
# (2026-07-24), so drop them from the prefetched cache.
metrics_dir = Path(os.environ["HF_MODULES_CACHE"]) / "evaluate_modules" / "metrics"
for entry in metrics_dir.glob("evaluate-comparison--*"):
    shutil.rmtree(entry, ignore_errors=True) if entry.is_dir() else entry.unlink(missing_ok=True)
    print(f"removed comparison variant from the cache: {entry.name}", flush=True)

from lm_eval.tasks import TaskManager, get_task_dict  # noqa: E402


# The task groups of scripts/evaluate_english-vllm.sh (swallow v202411).
# Constructing a task downloads its dataset into HF_DATASETS_CACHE.
TASKS = (
    "triviaqa",
    "gsm8k",
    "openbookqa",
    "hellaswag",
    "xwinograd_en",
    "squadv2",
    "mmlu",
    "bbh_cot_fewshot",
    "math_500",
    # Gated dataset: requires HF_TOKEN and an approved access request.
    "gpqa_main_cot_zeroshot_meta_llama3_wo_chat",
)

task_manager = TaskManager()
failed = []
for task in TASKS:
    try:
        get_task_dict([task], task_manager)
        print(f"prefetched task dataset: {task}", flush=True)
    except Exception:
        traceback.print_exc()
        failed.append(task)

if failed:
    print(
        f"WARNING: failed to prefetch {failed}; these tasks will need Hub access "
        "(and credentials for gated datasets) at evaluation time.",
        file=sys.stderr,
        flush=True,
    )
