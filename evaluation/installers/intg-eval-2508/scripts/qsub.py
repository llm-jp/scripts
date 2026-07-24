#!/usr/bin/env python3
"""Generate and submit a PBS qsub script for running swallow and llm-jp-eval.

Typical usage:
    python3 qsub.py <model_name_or_path> <output_dir>

Defaults:
    With no optional flags, runs swallow v202411 and both llm-jp-eval versions
    (v1.4.1 and v2.1.0). Use --disable-swallow and/or --disable-llm-jp-eval to
    skip each evaluation; use --llm-jp-eval-versions to select specific versions
    (v2.1.3 and v2.1.5, the latest release, are also available).

vllm-serve mode (EXPERIMENTAL):
    With --vllm-serve, all evaluations run against a single shared vLLM
    server, so the model is loaded once per job instead of once per lm_eval
    invocation / llm-jp-eval version (useful for large models):
        python3 qsub.py <model> <output_dir> --vllm-serve \\
            --llm-jp-eval-versions v2.1.5
    Requires the vllm-serve scripts installed under
    <experiment-dir>/environment/vllm-serve. --reasoning-parser is not yet
    implemented in the serve client, and v1.4.1 does not support
    --apply-chat-template. Note that scores follow the vLLM version of the
    *server* venv, so they are only comparable with offline runs that used
    the same vLLM version.

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
#PBS -o {output_dir}/logs/{artifact_stem}.out
#PBS -e {output_dir}/logs/{artifact_stem}.err
{options}

set -ux

# FIXME: Model name or **absolute path** to the model
MODEL_NAME_OR_PATH="{model_name_or_path}"
# FIXME: **Absolute path** to the output directory
OUTPUT_DIR="{output_dir}"

LOG_DIR="${{OUTPUT_DIR}}/logs"
mkdir -p $LOG_DIR

export HF_HOME="{hf_home}"
export HF_TOKEN="{hf_token}"{judge_env_lines}
export NUM_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPU-1)))
export NUMEXPR_MAX_THREADS=192

pushd {experiment_dir}/environment

{swallow_template}

{llm_jp_eval_template}

{llm_jp_judge_template}
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

LLM_JP_EVAL_TEMPLATE_LEGACY = """\
# Run llm-jp-eval {llm_jp_eval_version}
pushd llm-jp-eval-{llm_jp_eval_version}/
mkdir -p $OUTPUT_DIR/{llm_jp_eval_output_subdir}
bash run_llm-jp-eval.sh \\
    $MODEL_NAME_OR_PATH \\
    $OUTPUT_DIR/{llm_jp_eval_output_subdir} \\
    {max_num_samples} > $LOG_DIR/llm-jp-eval-{llm_jp_eval_version}.log 2> $LOG_DIR/llm-jp-eval-{llm_jp_eval_version}.err
popd
"""

LLM_JP_EVAL_TEMPLATE = """\
# Run llm-jp-eval {llm_jp_eval_version}
pushd llm-jp-eval-{llm_jp_eval_version}/
mkdir -p $OUTPUT_DIR/{llm_jp_eval_output_subdir}
LLM_JP_EVAL_OPTS=(--max_num_samples {max_num_samples}{apply_chat_template}{reasoning_parser}{chat_template_args}{basemodel})
bash run_llm-jp-eval.sh \\
    $MODEL_NAME_OR_PATH \\
    $OUTPUT_DIR/{llm_jp_eval_output_subdir} \\
    "${{LLM_JP_EVAL_OPTS[@]}}" > $LOG_DIR/llm-jp-eval-{llm_jp_eval_version}.log 2> $LOG_DIR/llm-jp-eval-{llm_jp_eval_version}.err
popd
"""

VLLM_SERVE_TEMPLATE = """\
# Run all evaluations against a single shared vLLM server (the model is
# loaded exactly once per job). Requires the vllm-serve scripts to be
# installed under {experiment_dir}/environment/vllm-serve (see
# installers/intg-eval-2508/vllm-serve/README.md).
bash {experiment_dir}/environment/vllm-serve/run_eval_serve.sh \\
    $MODEL_NAME_OR_PATH \\
    $OUTPUT_DIR \\
{serve_args}
"""

LLM_JP_JUDGE_TEMPLATE = """\
# Run llm-jp-judge (LLM-as-a-Judge). Generation runs on a local vLLM server;
# judging uses the configured judge client (see llm-jp-judge/README.md).
pushd llm-jp-judge/
LLM_JP_JUDGE_OPTS=({judge_opts})
bash run_llm-jp-judge.sh \\
    $MODEL_NAME_OR_PATH \\
    $OUTPUT_DIR/llm-jp-judge \\
    "${{LLM_JP_JUDGE_OPTS[@]}}" > $LOG_DIR/llm-jp-judge.log 2> $LOG_DIR/llm-jp-judge.err
popd
"""

# The same output subdirectory structure as qsub_nonbreaking.py
LLM_JP_EVAL_OUTPUT_SUBDIR_NONBREAKING = {
    "v1.4.1": "llm-jp-eval",
    "v2.1.0": "llm-jp-eval_v2.1.0",
    "v2.1.3": "llm-jp-eval_v2.1.3",
    "v2.1.5": "llm-jp-eval_v2.1.5",
}

# Versions whose run script takes option-style flags (--max_num_samples,
# --apply_chat_template, ...). v1.4.1 and v2.1.0 take positional arguments.
LLM_JP_EVAL_OPTS_STYLE_VERSIONS = ("v2.1.3", "v2.1.5")

# Versions usable with --vllm-serve. Every supported version has a dump /
# inference / eval split the serve-mode client plugs into (v1.4.x through
# run_llm-jp-eval-v1-serve.sh, v2.x through run_llm-jp-eval-serve.sh).
LLM_JP_EVAL_SERVE_VERSIONS = ("v1.4.1", "v2.1.0", "v2.1.3", "v2.1.5")

# Versions whose installer ships the base-model evaluation resources
# (config_basemodel.yaml / inference_config_basemodel.yaml) used by --basemodel.
LLM_JP_EVAL_BASEMODEL_VERSIONS = ("v2.1.5",)


def eval_output_targets(args):
    """Subdirectories under output_dir this job writes, and a slug naming them.

    Used to let separate jobs (e.g. a swallow-only job and an
    llm-jp-eval-only job) share one output directory without collisions.
    """
    targets = []
    parts = []
    if args.swallow_version and not args.disable_swallow:
        targets.append("swallow")
        parts.append("swallow")
    if args.llm_jp_eval_versions and not args.disable_llm_jp_eval:
        for version in args.llm_jp_eval_versions:
            if args.legacy_output:
                targets.append(LLM_JP_EVAL_OUTPUT_SUBDIR_NONBREAKING.get(version, f"llm-jp-eval_{version}"))
            else:
                targets.append(f"llm-jp-eval/{version}")
        parts.append("llm-jp-eval-" + "-".join(args.llm_jp_eval_versions))
    if args.llm_jp_judge:
        targets.append("llm-jp-judge")
        parts.append("llm-jp-judge")
    return targets, "+".join(parts) or "job"


def load_args():
    parser = argparse.ArgumentParser(description="Generate qsub script for evaluation jobs.")

    # General configuration
    parser.add_argument("model_name_or_path", type=str, help="Model name or absolute path to the model directory.")
    parser.add_argument("output_dir", type=str, help="Output directory for results.")
    parser.add_argument("--experiment-dir", type=str, default="/groups/gcg51557/experiments/0230_intg_eval_2509", help="Directory where the evaluation environment is located. Default is '/groups/gcg51557/experiments/0230_intg_eval_2509'.")

    # Evaluator versions
    parser.add_argument("--swallow-version", type=str, default="v202411", choices=["v202411", ""], help="Version of the swallow environment. If not specified, no swallow evaluation will be run.")
    parser.add_argument("--disable-swallow", action="store_true", help="Disable the swallow evaluation even if swallow_version is specified.")
    parser.add_argument("--llm-jp-eval-versions", type=str, nargs="+", default=["v1.4.1", "v2.1.0"], choices=["v1.4.1", "v2.1.0", "v2.1.3", "v2.1.5"], help="Versions of the llm-jp-eval environment to run.")
    parser.add_argument("--disable-llm-jp-eval", action="store_true", help="Disable the llm-jp-eval evaluation even if versions are specified.")

    # Job configuration
    parser.add_argument("--job-name", type=str, default="0195_intg_eval", help="Name of the job.")
    parser.add_argument("--rtype", type=str, default="rt_HG", choices=["rt_HG", "rt_HF"], help="Resource type for the job.")
    parser.add_argument("--select", type=int, default=1, help="Number of nodes (rt_HF) to use for the job.")
    parser.add_argument("--options", type=str, default=[], nargs="*", help="Additional options for the qsub script.")
    parser.add_argument("--dry-run", action="store_true", help="Print the generated qsub script and exit without submitting.")
    parser.add_argument("--legacy-output", action="store_true", help="Use the legacy output directory structure from /groups/gcg51557/evaluation/qsub.py (e.g. llm-jp-eval/ for v1.4.1, llm-jp-eval_v2.1.0/ for v2.1.0) instead of the default layout under llm-jp-eval/<version>/.")
    parser.add_argument("--pbs-queue", type=str, default="R9920261000", choices=["rt_HG", "rt_HF", "R9920261000"], help="PBS queue name (default: 'R9920261000').")
    parser.add_argument("--pbs-group", type=str, default="gcg51557", help="ABCI project group name for PBS '-P'.")

    # Resource configuration
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of tensor parallel groups.")
    parser.add_argument("--data-parallel-size", type=int, default=1, help="Number of data parallel groups.")
    parser.add_argument("--llm-jp-eval-max-num-samples", type=int, default=100, help="Maximum number of samples per dataset for llm-jp-eval. Set '-1' to use all samples.")
    parser.add_argument("--apply-chat-template", action="store_true", help="Apply chat template when running inference.")
    parser.add_argument("--reasoning-parser", type=str, default=None, help="Reasoning parser to extract final response (e.g. 'openai_gptoss').")
    parser.add_argument("--chat-template-args", type=str, nargs="*", default=None, metavar="KEY=VALUE", help="Extra keyword arguments for chat template application (e.g. 'reasoning_effort=low'). Requires --apply-chat-template.")
    parser.add_argument("--basemodel", action="store_true", help="Base-model (pretrained checkpoint) evaluation: fixed prompt template (config_basemodel.yaml), add_special_tokens=False, temperature=0.0 and the 4-shot datasets only (only_4shots.yaml). llm-jp-eval v2.1.5+ only; incompatible with --apply-chat-template. JA/EN scores are reported separately via lang_scores in result.json.")

    # llm-jp-judge (LLM-as-a-Judge)
    parser.add_argument("--llm-jp-judge", action="store_true", help="Run llm-jp-judge after the other evaluations. Requires the llm-jp-judge environment installed under <experiment-dir>/environment/llm-jp-judge. With --vllm-serve, generation reuses the shared server and judging runs after it is stopped.")
    parser.add_argument("--judge-client", type=str, default="openai", choices=["openai", "azure", "bedrock", "vllm"], help="Judge client for llm-jp-judge (default: openai). 'vllm' serves --judge-model locally (no external API; needed on clusters without outbound network access from compute nodes).")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-2024-08-06", help="Judge model name for llm-jp-judge (default: gpt-4o-2024-08-06).")
    parser.add_argument("--judge-base-url", type=str, default=None, help="Base URL for --judge-client openai (e.g. an OpenAI-compatible endpoint).")
    parser.add_argument("--judge-benchmark-size", type=int, default=None, help="Use only the first N samples of each llm-jp-judge benchmark (default: all samples).")
    parser.add_argument("--disable-mt-bench", action="store_true", help="Skip mt_bench_en / mt_bench_ja in llm-jp-judge.")

    # vllm-serve mode (EXPERIMENTAL)
    parser.add_argument("--vllm-serve", action="store_true", help="Run all evaluations against a single shared vLLM server so the model is loaded once per job (useful for large models). Requires the vllm-serve scripts installed under <experiment-dir>/environment/vllm-serve. --reasoning-parser is not yet implemented in the serve client. Scores follow the vLLM version of the server venv.")
    parser.add_argument("--serve-venv", type=str, default=None, help="venv that provides `vllm serve` (only with --vllm-serve; default: auto-detect, see vllm-serve/README.md).")
    parser.add_argument("--max-model-len", type=int, default=None, help="--max-model-len for the vLLM server (only with --vllm-serve; default: model config).")
    parser.add_argument("--client-concurrency", type=int, default=None, help="Prompts each evaluation client keeps in flight against the shared vLLM server (only with --vllm-serve; default: 256, on the order of vLLM's default max_num_seqs). Each client translates this into its own request shape, so the saturation target is framework-independent.")

    # Logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    return parser.parse_args()


def check_args(args):
    if not os.path.isabs(args.output_dir):
        raise ValueError(f"Output directory '{args.output_dir}' must be an absolute path.")

    if Path(args.output_dir).exists():
        targets, _ = eval_output_targets(args)
        existing = [t for t in targets if (Path(args.output_dir) / t).exists()]
        if existing:
            raise ValueError(
                f"Output directory '{args.output_dir}' already contains results for {existing}. "
                "Sharing an output directory between jobs is only possible when their "
                "evaluation targets do not overlap."
            )

    if args.rtype == "rt_HG" and args.select != 1:
        raise ValueError(f"Invalid selection '{args.select}' for resource type '{args.rtype}'. Only 1 GPU can be selected.")

    if not args.llm_jp_judge:
        if args.judge_base_url or args.judge_benchmark_size or args.disable_mt_bench:
            raise ValueError("--judge-base-url, --judge-benchmark-size and --disable-mt-bench require --llm-jp-judge.")
    else:
        if args.judge_client == "openai" and not (os.environ.get("OPENAI_API_KEY") or args.judge_base_url):
            logging.warning("OPENAI_API_KEY is not set; the judge phase will fail unless credentials are provided via a .env file in the llm-jp-judge checkout.")
        if args.judge_client == "azure" and not os.environ.get("AZURE_OPENAI_API_KEY"):
            logging.warning("AZURE_OPENAI_API_KEY is not set; the judge phase will fail unless credentials are provided via a .env file in the llm-jp-judge checkout.")

    if args.basemodel and not args.disable_llm_jp_eval:
        if args.apply_chat_template or args.chat_template_args:
            raise ValueError("--basemodel cannot be combined with --apply-chat-template / --chat-template-args.")
        unsupported = [v for v in args.llm_jp_eval_versions if v not in LLM_JP_EVAL_BASEMODEL_VERSIONS]
        if unsupported:
            raise ValueError(
                f"--basemodel supports llm-jp-eval versions {list(LLM_JP_EVAL_BASEMODEL_VERSIONS)} only, "
                f"got {unsupported}. Pass e.g. '--llm-jp-eval-versions v2.1.5'."
            )

    if args.vllm_serve:
        if not args.disable_llm_jp_eval:
            unsupported = [v for v in args.llm_jp_eval_versions if v not in LLM_JP_EVAL_SERVE_VERSIONS]
            if unsupported:
                raise ValueError(
                    f"--vllm-serve supports llm-jp-eval versions {list(LLM_JP_EVAL_SERVE_VERSIONS)} only, "
                    f"got {unsupported}. Pass e.g. '--llm-jp-eval-versions v2.1.5' (or --disable-llm-jp-eval)."
                )
            if "v1.4.1" in args.llm_jp_eval_versions and (args.apply_chat_template or args.chat_template_args):
                raise ValueError(
                    "--apply-chat-template / --chat-template-args are not supported by "
                    "llm-jp-eval v1.4.1 in serve mode."
                )
        if args.reasoning_parser:
            raise ValueError(
                "--reasoning-parser is not yet implemented in the serve-mode client "
                "(inference_openai.py); use the offline mode."
            )
    elif args.serve_venv or args.max_model_len or args.client_concurrency:
        raise ValueError("--serve-venv, --max-model-len and --client-concurrency require --vllm-serve.")


def main():
    args = load_args()

    logging.getLogger().setLevel(logging.INFO)

    check_args(args)

    tokenize_kwargs_json = ""
    if args.chat_template_args:
        # Default value hardcoded in llm-jp-eval-inference's BaseInferenceConfig.tokenize_kwargs
        d = {"add_special_tokens": True}
        d.update(dict(arg.split("=", 1) for arg in args.chat_template_args))
        tokenize_kwargs_json = json.dumps(d, ensure_ascii=False, separators=(",", ":"))

    swallow_template = ""
    llm_jp_eval_template = ""
    llm_jp_judge_template = ""
    if args.llm_jp_judge and not args.vllm_serve:
        judge_opts = [
            f"--judge-client {args.judge_client}",
            f"--judge-model {args.judge_model}",
            f"--tensor-parallel-size {args.tensor_parallel_size}",
            f"--gpu-memory-utilization {args.gpu_memory_utilization}",
        ]
        if args.judge_base_url:
            judge_opts.append(f"--judge-base-url {args.judge_base_url}")
        if args.judge_benchmark_size:
            judge_opts.append(f"--benchmark-size {args.judge_benchmark_size}")
        if args.disable_mt_bench:
            judge_opts.append("--disable-mt-bench")
        llm_jp_judge_template = LLM_JP_JUDGE_TEMPLATE.format(judge_opts=" ".join(judge_opts))
    if args.vllm_serve:
        serve_args = [f"--experiment-dir {args.experiment_dir}"]
        if args.serve_venv:
            serve_args.append(f"--serve-venv {args.serve_venv}")
        serve_args.append(f"--tensor-parallel-size {args.tensor_parallel_size}")
        if args.data_parallel_size != 1:
            serve_args.append(f"--data-parallel-size {args.data_parallel_size}")
        serve_args.append(f"--gpu-memory-utilization {args.gpu_memory_utilization}")
        if args.max_model_len:
            serve_args.append(f"--max-model-len {args.max_model_len}")
        if args.client_concurrency:
            serve_args.append(f"--client-concurrency {args.client_concurrency}")
        if args.swallow_version and not args.disable_swallow:
            serve_args.append(f"--swallow --swallow-env swallow_{args.swallow_version}")
        if args.llm_jp_eval_versions and not args.disable_llm_jp_eval:
            serve_args.append("--llm-jp-eval-versions " + " ".join(args.llm_jp_eval_versions))
            serve_args.append(f"--max-num-samples {args.llm_jp_eval_max_num_samples}")
            if args.apply_chat_template:
                serve_args.append("--apply-chat-template")
            if tokenize_kwargs_json:
                serve_args.append(f"--tokenize-kwargs '{tokenize_kwargs_json}'")
            if args.basemodel:
                serve_args.append("--basemodel")
            if args.legacy_output:
                serve_args.append("--legacy-output")
        if args.llm_jp_judge:
            serve_args.append(f"--llm-jp-judge --judge-client {args.judge_client} --judge-model {args.judge_model}")
            if args.judge_base_url:
                serve_args.append(f"--judge-base-url {args.judge_base_url}")
            if args.judge_benchmark_size:
                serve_args.append(f"--judge-benchmark-size {args.judge_benchmark_size}")
            if args.disable_mt_bench:
                serve_args.append("--disable-mt-bench")
        swallow_template = VLLM_SERVE_TEMPLATE.format(
            experiment_dir=args.experiment_dir,
            serve_args=" \\\n".join(f"    {a}" for a in serve_args),
        )
    if not args.vllm_serve and args.swallow_version and not args.disable_swallow:
        swallow_template = SWALLOW_TEMPLATE.format(
            swallow_version=args.swallow_version,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            data_parallel_size=args.data_parallel_size,
        )
    if not args.vllm_serve and args.llm_jp_eval_versions and not args.disable_llm_jp_eval:
        apply_chat_template_flag = " --apply_chat_template" if args.apply_chat_template else ""
        reasoning_parser_flag = f" --reasoning_parser {args.reasoning_parser}" if args.reasoning_parser else ""
        chat_template_args_flag = f" --tokenize_kwargs '{tokenize_kwargs_json}'" if tokenize_kwargs_json else ""
        basemodel_flag = " --basemodel" if args.basemodel else ""
        chunks = []
        for version in args.llm_jp_eval_versions:
            if args.legacy_output:
                llm_jp_eval_output_subdir = LLM_JP_EVAL_OUTPUT_SUBDIR_NONBREAKING.get(
                    version, f"llm-jp-eval_{version}"
                )
            else:
                llm_jp_eval_output_subdir = f"llm-jp-eval/{version}"
            if version in LLM_JP_EVAL_OPTS_STYLE_VERSIONS:
                chunk = LLM_JP_EVAL_TEMPLATE.format(
                    llm_jp_eval_version=version,
                    llm_jp_eval_output_subdir=llm_jp_eval_output_subdir,
                    max_num_samples=args.llm_jp_eval_max_num_samples,
                    apply_chat_template=apply_chat_template_flag,
                    reasoning_parser=reasoning_parser_flag,
                    chat_template_args=chat_template_args_flag,
                    basemodel=basemodel_flag,
                )
            else:
                chunk = LLM_JP_EVAL_TEMPLATE_LEGACY.format(
                    llm_jp_eval_version=version,
                    llm_jp_eval_output_subdir=llm_jp_eval_output_subdir,
                    max_num_samples=args.llm_jp_eval_max_num_samples,
                )
            chunks.append(chunk)
        llm_jp_eval_template = "\n".join(chunks)

    # When sharing an existing output directory with other jobs, suffix the
    # job artifacts (script, config, PBS logs) so they do not overwrite each
    # other; a fresh directory keeps the historical names.
    _, eval_slug = eval_output_targets(args)
    if os.path.exists(args.output_dir):
        artifact_stem = f"qsub_{eval_slug}"
        config_name = f"config_{eval_slug}.json"
    else:
        artifact_stem = "qsub"
        config_name = "config.json"

    hf_home = os.environ.get("HF_HOME")

    if not hf_home:
        raise ValueError("HF_HOME environment variable is not set. Please set it to the path where Hugging Face datasets are stored.")

    hf_home = hf_home.strip()
    if not hf_home.startswith("/groups/gcg51557/experiments"):
        raise ValueError(f"HF_HOME '{hf_home}' must be within the '/groups/gcg51557/experiments' directory.")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set. Please set it to your Hugging Face token.")

    # Forward judge API credentials into the job script (compute nodes do not
    # inherit the submission environment).
    judge_env_lines = ""
    if args.llm_jp_judge:
        judge_env_vars = (
            "OPENAI_API_KEY", "OPENAI_BASE_URL",
            "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "OPENAI_API_VERSION",
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION",
        )
        for name in judge_env_vars:
            value = os.environ.get(name)
            if value:
                judge_env_lines += f'\nexport {name}="{value}"'

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
        artifact_stem=artifact_stem,
        pbs_group=args.pbs_group,
        swallow_version=args.swallow_version,
        llm_jp_eval_versions=args.llm_jp_eval_versions,
        swallow_template=swallow_template,
        llm_jp_eval_template=llm_jp_eval_template,
        llm_jp_judge_template=llm_jp_judge_template,
        judge_env_lines=judge_env_lines,
        pbs_queue=args.pbs_queue,
    )

    if args.dry_run:
        print(qsub_script)
        return

    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    qsub_script_path = os.path.join(args.output_dir, f"{artifact_stem}.sh")
    with open(qsub_script_path, "w") as f:
        f.write(qsub_script)

    config_path = os.path.join(args.output_dir, config_name)
    config = dict(args._get_kwargs())
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # 実際にジョブを投げる
    result = subprocess.run(["qsub", qsub_script_path], check=True, stdout=subprocess.PIPE, text=True)
    job_id = result.stdout.strip()
    logging.info(f"JOB ID: {job_id}")


if __name__ == "__main__":
    main()
