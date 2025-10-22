# ABCI qsub scripts

This directory hosts the ABCI job submission helpers that we distribute under
`/groups/gcg51557/evaluation*`.

- `qsub.py` is the refreshed script aligned with the new evaluation environment
  at `/groups/gcg51557/experiments/0230_intg_eval_2509`. It can run swallow and
  multiple llm-jp-eval versions in one job. Because it reorganizes the working
  directories, it introduces breaking changes for users of the legacy workflow,
  so we now mirror this file to `/groups/gcg51557/evaluation2/qsub.py`.
- `qsub_nonbreaking.py` is a compatibility branch of the long-lived script that
  used to live at `/groups/gcg51557/evaluation/qsub.py`. It keeps the original
  behaviour (for example the default experiment directory
  `/groups/gcg51557/experiments/0182_intg_eval_2507` and the old shell flow) so
  existing automation keeps working. The only intended change is the minimal
  logic required to run llm-jp-eval v2.1.0 without forcing any of the breaking
  updates introduced in `qsub.py`.

Use `qsub.py` when you can adopt the new directory layout, and fall back to
`qsub_nonbreaking.py` when you must stay compatible with the legacy setup while
still running llm-jp-eval v2.1.0.
