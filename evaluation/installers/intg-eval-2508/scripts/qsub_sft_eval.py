#!/usr/bin/env python3
"""Submit a lightweight SFT job (llm-jp/simple_tuning) followed by an intg-eval
job that evaluates the resulting checkpoint (PBS / ABCI).

This is a thin orchestrator around two existing submitters:

  1. ``<simple-tuning-dir>/qsub_sft.py`` runs HF -> NeMo -> SFT -> HF in one
     PBS job and writes the final checkpoint to ``<output_dir>/sft/converted/final_hf``.
  2. ``qsub.py`` (this directory) evaluates that checkpoint. It is submitted
     right away with ``#PBS -W depend=afterok:<sft_job_id>`` so it starts only
     after the SFT job succeeded (PBS deletes it if the SFT job fails).

The evaluation flags needed for a simple_tuning output (Harmony chat template,
gpt-oss reasoning parser, llm-jp-judge settings for thinking models) are fixed
here so they do not have to be retyped. Anything after ``--`` is forwarded to
qsub.py verbatim, so qsub.py itself does not grow SFT-specific options.

Typical usage:
    python3 qsub_sft_eval.py \\
        /groups/gcg51557/experiments/0297_v4-8b-phase2/tasks/decay4t/checkpoints_hf/iter_0500000 \\
        /groups/gcg51557/experiments/<your_exp>/results/sft-eval-8b \\
        --param-name llmjp4_8b_4K \\
        -- --judge-model llm-jp-4-32b-a3b-thinking --pbs-queue rt_HG
"""

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_SIMPLE_TUNING_DIR = "/groups/gcg51557/experiments/0366_simple_tuning/simple_tuning"
SFT_CONFIG_NAME = "sft_simple"
# The submitting user usually has no wandb credentials on ABCI; a user-supplied
# --sft-override for the same key comes later on the command line and wins.
SFT_DEFAULT_OVERRIDES = ["exp_manager.create_wandb_logger=False"]

# Evaluation preset for a simple_tuning checkpoint (llm-jp-4, Harmony chat
# template). See VALIDATION.md (2026-07-31) for the llm-jp-judge thinking-model
# settings. The reasoning effort is shared between llm-jp-eval and the judge
# generation and matches the "reasoning_low" datasets used by sft_simple.
# The context length is left to the checkpoint's config.json (4096 for the
# *_4K parameter sets; vLLM refuses a larger --max-model-len), so the judge
# generation budget is derived from the parameter set's context suffix below.
# The reasoning parser is llm-jp-eval-inference's ``llmjp4`` adapter (v2.1.5,
# commit c6cd0fa): llm-jp-4 emits Harmony channels with its own vocabulary, so
# vLLM's ``openai_gptoss`` parser applied to its token IDs yields empty output.
EVAL_PRESET = [
    "--llm-jp-eval-versions", "v2.1.5",
    "--apply-chat-template",
    "--reasoning-parser", "llmjp4",
    "--llm-jp-judge",
    "--judge-gen-extract-final",
]
JUDGE_GEN_MAX_TOKENS_CAP = 8192


def judge_gen_max_tokens(param_name: str) -> str | None:
    """Generation budget for llm-jp-judge: half of the context implied by the
    simple_tuning parameter set name (``..._4K`` -> 4096 -> 2048), capped at
    JUDGE_GEN_MAX_TOKENS_CAP. Prompt + max_tokens must fit in the context or
    vLLM rejects the request. None (llm-jp-judge's per-benchmark default) when
    the name carries no context suffix."""
    match = re.search(r"_(\d+)K$", param_name)
    if not match:
        return None
    return str(min(JUDGE_GEN_MAX_TOKENS_CAP, int(match.group(1)) * 1024 // 2))


def load_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a simple_tuning SFT job and a dependent intg-eval job (ABCI).",
        epilog="Arguments after '--' are forwarded to qsub.py verbatim (e.g. -- --judge-model <name> --pbs-queue rt_HG).",
    )
    parser.add_argument("input_hf_path", type=str, help="Absolute path to the base (pretrained) Hugging Face checkpoint.")
    parser.add_argument("output_dir", type=str, help="Absolute output directory. SFT artifacts go to <output_dir>/sft, evaluation results to <output_dir>/eval.")

    # SFT (simple_tuning) configuration
    parser.add_argument("--simple-tuning-dir", type=str, default=os.environ.get("SIMPLE_TUNING_DIR", DEFAULT_SIMPLE_TUNING_DIR), help=f"Checkout of llm-jp/simple_tuning that provides qsub_sft.py (default: $SIMPLE_TUNING_DIR or {DEFAULT_SIMPLE_TUNING_DIR}).")
    parser.add_argument("--param-name", type=str, default="llmjp4_8b_4K", help="simple_tuning parameter file name (scripts/abci/train/params/sft/<name>.sh), e.g. llmjp4_8b_4K, llmjp4_32b-a3b_4K, llmjp4_32b_4K (default: llmjp4_8b_4K).")
    parser.add_argument("--sft-num-nodes", type=int, default=1, help="rt_HF nodes for the SFT job (default: 1; the 32b configs use 2).")
    parser.add_argument("--sft-walltime", type=str, default="10:00:00", help="Walltime of the SFT job (default: 10:00:00).")
    parser.add_argument("--sft-override", dest="sft_overrides", type=str, nargs="*", default=[], metavar="KEY=VALUE", help=f"Extra hydra overrides for train_sft.py (appended after the defaults {SFT_DEFAULT_OVERRIDES}; e.g. trainer.sft.max_steps=20 for a smoke test).")

    # Evaluation configuration
    parser.add_argument("--reasoning-effort", type=str, default="low", choices=["low", "medium", "high"], help="reasoning_effort for the SFT model in llm-jp-eval (chat template) and llm-jp-judge generation (default: low).")

    # Shared PBS configuration (kept in sync between the two jobs)
    parser.add_argument("--job-name", type=str, default="0195_intg_eval", help="PBS job name for both jobs.")
    parser.add_argument("--pbs-queue", type=str, default="R9920261000", choices=["rt_HF", "R9920261000"], help="PBS queue for the SFT job (default: R9920261000). The evaluation job uses qsub.py's default unless --pbs-queue is forwarded after '--'.")
    parser.add_argument("--pbs-group", type=str, default="gcg51557", help="ABCI project group for both jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Print the two submission commands and the rendered evaluation job script without submitting anything.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Split at the first '--': everything before is ours, everything after is
    # forwarded to qsub.py untouched (argparse.REMAINDER is not reliable here).
    argv = sys.argv[1:]
    own_args, eval_args = (argv[: argv.index("--")], argv[argv.index("--") + 1 :]) if "--" in argv else (argv, [])
    args = parser.parse_args(own_args)
    args.eval_args = eval_args
    return args


def check_args(args: argparse.Namespace) -> None:
    for attr in ("input_hf_path", "output_dir", "simple_tuning_dir"):
        value = getattr(args, attr)
        if not os.path.isabs(value):
            raise ValueError(f"{attr.replace('_', '-')} must be an absolute path: {value}")
    if args.sft_num_nodes <= 0:
        raise ValueError("--sft-num-nodes must be positive.")
    for override in args.sft_overrides:
        if "=" not in override:
            raise ValueError(f"--sft-override entries must look like KEY=VALUE: {override}")
    if args.eval_args and not args.eval_args[0].startswith("-"):
        raise ValueError(f"Forwarded qsub.py arguments must start with an option (model and output directory are set by this script): {args.eval_args}")
    if not args.dry_run and not (Path(args.simple_tuning_dir) / "qsub_sft.py").is_file():
        raise ValueError(f"qsub_sft.py not found under --simple-tuning-dir: {args.simple_tuning_dir}")


def build_sft_command(args: argparse.Namespace, sft_dir: Path, run_name: str, dry_run: bool) -> list[str]:
    cmd = [
        sys.executable, str(Path(args.simple_tuning_dir) / "qsub_sft.py"),
        "--run-name", run_name,
        "--param-name", args.param_name,
        "--input-hf-path", args.input_hf_path,
        "--output-dir", str(sft_dir),
        "--sft-config-name", SFT_CONFIG_NAME,
        "--pbs-job-name", args.job_name,
        "--num-nodes", str(args.sft_num_nodes),
        "--pbs-queue", args.pbs_queue,
        "--pbs-group", args.pbs_group,
        "--walltime", args.sft_walltime,
        "--override", *SFT_DEFAULT_OVERRIDES, *args.sft_overrides,
    ]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def build_eval_command(args: argparse.Namespace, model_path: Path, eval_dir: Path, sft_job_id: str, dry_run: bool) -> list[str]:
    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / "qsub.py"),
        str(model_path), str(eval_dir),
        "--job-name", args.job_name,
        "--pbs-group", args.pbs_group,
        # nargs="*" options: keep the next token an option so it is not swallowed.
        "--options", f"#PBS -W depend=afterok:{sft_job_id}",
        *EVAL_PRESET,
        "--chat-template-args", f"reasoning_effort={args.reasoning_effort}",
        "--judge-gen-reasoning-effort", args.reasoning_effort,
    ]
    gen_max_tokens = judge_gen_max_tokens(args.param_name)
    if gen_max_tokens is not None:
        cmd += ["--judge-gen-max-tokens", gen_max_tokens]
    cmd += args.eval_args
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def preflight_eval(cmd: list[str], show_output: bool) -> None:
    """Render the evaluation job with qsub.py --dry-run to validate its arguments."""
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if show_output or result.returncode != 0:
        sys.stdout.write(result.stdout)
        sys.stdout.flush()
    if result.returncode != 0:
        raise ValueError("qsub.py rejected the evaluation arguments (see the output above); nothing was submitted.")


def run_capture(cmd: list[str]) -> str:
    """Run a submitter, echo its output, and return the combined stdout/stderr."""
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sys.stdout.write(result.stdout)
    sys.stdout.flush()
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {display(cmd)}")
    return result.stdout


def display(cmd: list[str]) -> str:
    """Shell-quoted command for humans: the interpreter and this directory's
    qsub.py are shown by name so the output does not depend on the machine."""
    shown = ["python3" if cmd[0] == sys.executable else cmd[0]]
    shown += ["qsub.py" if tok == str(Path(__file__).resolve().parent / "qsub.py") else tok for tok in cmd[1:]]
    return shlex.join(shown)


def parse_job_id(output: str) -> str:
    match = re.search(r"JOB ID:\s*(\S+)", output)
    if not match:
        raise RuntimeError("Could not find 'JOB ID: <id>' in the submitter output.")
    return match.group(1)


def main() -> None:
    args = load_args()
    check_args(args)

    output_dir = Path(args.output_dir)
    sft_dir = output_dir / "sft"
    eval_dir = output_dir / "eval"
    model_path = sft_dir / "converted" / "final_hf"
    run_name = output_dir.name

    sft_cmd = build_sft_command(args, sft_dir, run_name, dry_run=False)

    if args.dry_run:
        print("# [1/2] SFT job (simple_tuning):")
        print(display(sft_cmd))
        print()
        print("# [2/2] Evaluation job (qsub.py), submitted with a dependency on the SFT job:")
        print(display(build_eval_command(args, model_path, eval_dir, "<SFT_JOB_ID>", dry_run=False)))
        print()
        print("# Rendered evaluation job script:")
        preflight_eval(build_eval_command(args, model_path, eval_dir, "<SFT_JOB_ID>", dry_run=True), show_output=True)
        return

    # Preflight: validate the evaluation arguments (HF_HOME/HF_TOKEN, option
    # combinations) before anything is submitted, so a rejected evaluation
    # command cannot leave an orphan SFT job behind.
    preflight_eval(build_eval_command(args, model_path, eval_dir, "<SFT_JOB_ID>", dry_run=True), show_output=False)

    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Submitting the SFT job: %s", display(sft_cmd))
    sft_job_id = parse_job_id(run_capture(sft_cmd))

    eval_cmd = build_eval_command(args, model_path, eval_dir, sft_job_id, dry_run=False)
    logging.info("Submitting the evaluation job: %s", display(eval_cmd))
    eval_job_id = parse_job_id(run_capture(eval_cmd))

    record = {
        "args": dict(args._get_kwargs()),
        "sft_job_id": sft_job_id,
        "sft_command": sft_cmd,
        "eval_job_id": eval_job_id,
        "eval_command": eval_cmd,
        "model_path": str(model_path),
    }
    (output_dir / "qsub_sft_eval.json").write_text(json.dumps(record, indent=4, ensure_ascii=False))
    logging.info("SFT JOB ID: %s / EVAL JOB ID: %s (depend=afterok)", sft_job_id, eval_job_id)


if __name__ == "__main__":
    main()
