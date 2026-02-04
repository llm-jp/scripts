#!/usr/bin/env python3
"""Generate and submit a PBS qsub script for running swallow and llm-jp-eval.

Typical usage:
    python3 qsub.py <model_name_or_path> <output_dir>

Defaults:
    With no optional flags, runs swallow v202411 and both llm-jp-eval versions
    (v1.4.1 and v2.1.0). Use --disable-swallow and/or --disable-llm-jp-eval to
    skip each evaluation; use --llm-jp-eval-versions to select specific versions.

Environment variables:
    HF_HOME must point to a path under /groups/gcg51557/experiments.
    HF_TOKEN must be set for Hugging Face access.

Outputs:
    Swallow evaluation artifacts are written under <output_dir>/swallow (logs in <output_dir>/logs/).
    For each llm-jp-eval version, writes the final evaluation result to
    <output_dir>/llm-jp-eval/<version>/results/result.json (one file per version).
"""
import argparse
import json
import os
import logging
import subprocess

from pathlib import Path

TEMPLATE = """#!/bin/sh
#PBS -N {job_name}
#PBS -P {pbs_group}
#PBS -q {pbs_queue}
#PBS -v RTYPE={rtype}
#PBS -l select={select}
#PBS -l walltime=24:00:00
#PBS -o {output_dir}/logs/qsub.out
#PBS -e {output_dir}/logs/qsub.err
{options}

set -ux

# FIXME: Model name or **absolute path** to the model
MODEL_NAME_OR_PATH="{model_name_or_path}"
# FIXME: **Absolute path** to the output directory
OUTPUT_DIR="{output_dir}"

LOG_DIR="${{OUTPUT_DIR}}/logs"
mkdir -p $LOG_DIR

export HF_HOME="{hf_home}"
export HF_TOKEN="{hf_token}"
export NUM_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPU-1)))
export NUMEXPR_MAX_THREADS=192

pushd {experiment_dir}/environment

{swallow_template}

{llm_jp_eval_template}
"""

SWALLOW_TEMPLATE = """\
# Run swallow evaluation
pushd swallow_{swallow_version}/
bash run-eval.sh \\
    $MODEL_NAME_OR_PATH \\
    $OUTPUT_DIR/swallow \\
    {gpu_memory_utilization} \\
    {tensor_parallel_size} \\
    {data_parallel_size} \\
    > $LOG_DIR/swallow_eval.log 2> $LOG_DIR/swallow_eval.err
popd
"""

LLM_JP_EVAL_TEMPLATE = """\
# Run llm-jp-eval {llm_jp_eval_version}
pushd llm-jp-eval-{llm_jp_eval_version}/
mkdir -p $OUTPUT_DIR/llm-jp-eval/{llm_jp_eval_version}
bash run_llm-jp-eval.sh \\
    $MODEL_NAME_OR_PATH \\
    $OUTPUT_DIR/llm-jp-eval/{llm_jp_eval_version} \\
    {max_num_samples} > $LOG_DIR/llm-jp-eval-{llm_jp_eval_version}.log 2> $LOG_DIR/llm-jp-eval-{llm_jp_eval_version}.err
popd
"""

def load_args():
    parser = argparse.ArgumentParser(description="Generate qsub script for evaluation jobs.")

    # General configuration
    parser.add_argument("model_name_or_path", type=str, help="Model name or absolute path to the model directory.")
    parser.add_argument("output_dir", type=str, help="Output directory for results.")
    parser.add_argument("--experiment-dir", type=str, default="/groups/gcg51557/experiments/0230_intg_eval_2509", help="Directory where the evaluation environment is located. Default is '/groups/gcg51557/experiments/0230_intg_eval_2509'.")

    # Evaluator versions
    parser.add_argument("--swallow-version", type=str, default="v202411", choices=["v202411", ""], help="Version of the swallow environment. If not specified, no swallow evaluation will be run.")
    parser.add_argument("--disable-swallow", action="store_true", help="Disable the swallow evaluation even if swallow_version is specified.")
    parser.add_argument("--llm-jp-eval-versions", type=str, nargs="+", default=["v1.4.1", "v2.1.0"], choices=["v1.4.1", "v2.1.0"], help="Versions of the llm-jp-eval environment to run.")
    parser.add_argument("--disable-llm-jp-eval", action="store_true", help="Disable the llm-jp-eval evaluation even if versions are specified.")

    # Job configuration
    parser.add_argument("--job-name", type=str, default="0195_intg_eval", help="Name of the job.")
    parser.add_argument("--rtype", type=str, default="rt_HG", choices=["rt_HG", "rt_HF"], help="Resource type for the job.")
    parser.add_argument("--select", type=int, default=1, help="Number of nodes (rt_HF) to use for the job.")
    parser.add_argument("--options", type=str, default=[], nargs="*", help="Additional options for the qsub script.")
    parser.add_argument("--dry-run", action="store_true", help="Print the generated qsub script and exit without submitting.")
    parser.add_argument("--pbs-queue", type=str, default="R9920251000", choices=["rt_HG", "rt_HF", "R9920251000"], help="PBS queue name (default: 'R9920251000').")
    parser.add_argument("--pbs-group", type=str, default="gcg51557", help="ABCI project group name for PBS '-P'.")

    # Resource configuration
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of tensor parallel groups.")
    parser.add_argument("--data-parallel-size", type=int, default=1, help="Number of data parallel groups.")
    parser.add_argument("--llm-jp-eval-max-num-samples", type=int, default=100, help="Maximum number of samples per dataset for llm-jp-eval.")

    # Logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    return parser.parse_args()


def check_args(args):
    if not os.path.isabs(args.output_dir):
        raise ValueError(f"Output directory '{args.output_dir}' must be an absolute path.")

    if Path(args.output_dir).exists():
        raise ValueError(f"Output directory '{args.output_dir}' already exists. Please specify a new path.")

    if args.rtype == "rt_HG" and args.select != 1:
        raise ValueError(f"Invalid selection '{args.select}' for resource type '{args.rtype}'. Only 1 GPU can be selected.")


def main():
    args = load_args()

    logging.getLogger().setLevel(logging.INFO)

    check_args(args)

    swallow_template = ""
    if args.swallow_version and not args.disable_swallow:
        swallow_template = SWALLOW_TEMPLATE.format(
            swallow_version=args.swallow_version,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            data_parallel_size=args.data_parallel_size,
        )
    llm_jp_eval_template = ""
    if args.llm_jp_eval_versions and not args.disable_llm_jp_eval:
        llm_jp_eval_template = "\n".join(
            LLM_JP_EVAL_TEMPLATE.format(llm_jp_eval_version=version, max_num_samples=args.llm_jp_eval_max_num_samples)
            for version in args.llm_jp_eval_versions
        )

    hf_home = os.environ.get("HF_HOME")

    if not hf_home:
        raise ValueError("HF_HOME environment variable is not set. Please set it to the path where Hugging Face datasets are stored.")

    hf_home = hf_home.strip()
    if not hf_home.startswith("/groups/gcg51557/experiments"):
        raise ValueError(f"HF_HOME '{hf_home}' must be within the '/groups/gcg51557/experiments' directory.")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set. Please set it to your Hugging Face token.")

    qsub_script = TEMPLATE.format(
        job_name=args.job_name,
        rtype=args.rtype,
        select=args.select,
        output_dir=args.output_dir,
        hf_home=hf_home,
        hf_token=hf_token,
        experiment_dir=args.experiment_dir,
        model_name_or_path=args.model_name_or_path,
        options="\n".join(args.options),
        pbs_group=args.pbs_group,
        swallow_version=args.swallow_version,
        llm_jp_eval_versions=args.llm_jp_eval_versions,
        swallow_template=swallow_template,
        llm_jp_eval_template=llm_jp_eval_template,
        pbs_queue=args.pbs_queue,
    )

    if args.dry_run:
        print(qsub_script)
        return

    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    qsub_script_path = os.path.join(args.output_dir, "qsub.sh")
    with open(qsub_script_path, "w") as f:
        f.write(qsub_script)

    config_path = os.path.join(args.output_dir, "config.json")
    config = dict(args._get_kwargs())
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # 実際にジョブを投げる
    result = subprocess.run(["qsub", qsub_script_path], check=True, stdout=subprocess.PIPE, text=True)
    job_id = result.stdout.strip()
    logging.info(f"JOB ID: {job_id}")


if __name__ == "__main__":
    main()
