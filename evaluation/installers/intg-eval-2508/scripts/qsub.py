#!/usr/bin/env python3
import argparse
import json
import os
import logging
import subprocess

from pathlib import Path

TEMPLATE = """#!/bin/sh
#PBS -N {job_name}
#PBS -P gcg51557
#PBS -q R9920251000
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
# Run llm-jp-eval
pushd llm-jp-eval-{llm_jp_eval_version}/
bash run_llm-jp-eval.sh \\
    $MODEL_NAME_OR_PATH \\
    $OUTPUT_DIR/llm-jp-eval > $LOG_DIR/llm-jp-eval.log 2> $LOG_DIR/llm-jp-eval.err
popd
"""

def load_args():
    parser = argparse.ArgumentParser(description="Generate qsub script for evaluation jobs.")

    # General configuration
    parser.add_argument("model_name_or_path", type=str, help="Model name or absolute path to the model directory.")
    parser.add_argument("output_dir", type=str, help="Output directory for results.")
    parser.add_argument("--experiment_dir", type=str, default="/groups/gcg51557/experiments/0230_intg_eval_2509", help="Directory where the evaluation environment is located. Default is '/groups/gcg51557/experiments/0230_intg_eval_2509'.")

    # Evaluator versions
    parser.add_argument("--swallow_version", type=str, default="v202411", choices=["v202411", ""], help="Version of the swallow environment. If not specified, no swallow evaluation will be run.")
    parser.add_argument("--llm_jp_eval_version", type=str, default="v1.4.1", choices=["v1.4.1", "v2.1.0"], help="Version of the llm-jp-eval environment. If not specified, no llm-jp-eval will be run.")

    # Job configuration
    parser.add_argument("--job_name", type=str, default="0195_intg_eval", help="Name of the job.")
    parser.add_argument("--rtype", type=str, default="rt_HG", choices=["rt_HG", "rt_HF"], help="Resource type for the job.")
    parser.add_argument("--select", type=int, default=1, help="Number of gpus (rt_HG) or nodes (rt_HF) to use for the job.")
    parser.add_argument("--options", type=str, default=[], nargs="*", help="Additional options for the qsub script.")

    # Resource configuration
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of tensor parallel groups.")
    parser.add_argument("--data_parallel_size", type=int, default=1, help="Number of data parallel groups.")

    # Logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    return parser.parse_args()


def check_args(args):
    if not os.path.isabs(args.output_dir):
        raise ValueError(f"Output directory '{args.output_dir}' must be an absolute path.")

    if args.rtype == "rt_HG" and args.select != 1:
        raise ValueError(f"Invalid selection '{args.select}' for resource type '{args.rtype}'. Only 1 GPU can be selected.")


def main():
    args = load_args()

    logging.getLogger().setLevel(logging.INFO)

    check_args(args)

    swallow_template = ""
    if args.swallow_version:
        swallow_template = SWALLOW_TEMPLATE.format(
            swallow_version=args.swallow_version,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            data_parallel_size=args.data_parallel_size,
        )
    llm_jp_eval_template = ""
    if args.llm_jp_eval_version:
        llm_jp_eval_template = LLM_JP_EVAL_TEMPLATE.format(llm_jp_eval_version=args.llm_jp_eval_version)

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
        swallow_version=args.swallow_version,
        llm_jp_eval_version=args.llm_jp_eval_version,
        swallow_template=swallow_template,
        llm_jp_eval_template=llm_jp_eval_template
    )

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
