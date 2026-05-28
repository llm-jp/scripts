#!/usr/bin/env python3
"""Generate and submit a PBS qsub script for training-batch replay.

Typical usage:
    python3 scripts/corpus/training-batch-replay/qsub.py \
        --task-dir /groups/gcg51557/experiments/0213_v4-8b/tasks/v4-8b \
        --env-dir /groups/gcg51557/experiments/0213_v4-8b/env \
        --output-dir /groups/gcg51557/experiments/0342_llm-jp-4-corpus-dump/replayed_batches/v4-8b \
        --iter-start 20000 --iter-end 40000 \
        --pbs-job-name 0342
"""

import argparse
import json
import logging
import os
import shlex
import subprocess
from pathlib import Path


DEFAULT_ENV_DIR = "/groups/gcg51557/experiments/0213_v4-8b/env"
DEFAULT_OUTPUT_DIR = "/groups/gcg51557/experiments/0342_llm-jp-4-corpus-dump/replayed_batches/v4-8b"
DEFAULT_TASK_DIR = "/groups/gcg51557/experiments/0213_v4-8b/tasks/v4-8b"


TEMPLATE = """#!/bin/bash
#PBS -N {job_name}
#PBS -P {pbs_group}
#PBS -q {pbs_queue}
#PBS -v RTYPE={resource_type}
#PBS -l select={select}:ncpus={ncpus}
#PBS -l walltime={walltime}
#PBS -o {output_dir}/logs/qsub_{range_label}.out
#PBS -e {output_dir}/logs/qsub_{range_label}.err
#PBS -m n
{pbs_options}

set -euxo pipefail
umask 002

REPO_DIR={repo_dir}
ENV_DIR={env_dir}
TASK_DIR={task_dir}
OUTPUT_DIR={output_dir}
OUTPUT={output}
ITER_INDEX={iter_index}
ITER_START={iter_start}
ITER_END={iter_end}
READER_WORKERS={reader_workers}
COMPRESS_WORKERS={compress_workers}
COMPRESS_CHUNK_RECORDS={compress_chunk_records}
GZIP_COMPRESSLEVEL={gzip_compresslevel}

mkdir -p "${{OUTPUT_DIR}}/logs"
chmod -R g+rwX "${{OUTPUT_DIR}}" || true

JOBID=${{PBS_JOBID%%.*}}
QSUB_LOG="${{OUTPUT_DIR}}/logs/qsub_{range_label}-${{JOBID}}.log"
exec > "${{QSUB_LOG}}" 2>&1

echo "JOBID=${{JOBID}}"
echo "REPO_DIR=${{REPO_DIR}}"
echo "ENV_DIR=${{ENV_DIR}}"
echo "TASK_DIR=${{TASK_DIR}}"
echo "OUTPUT_DIR=${{OUTPUT_DIR}}"
echo "OUTPUT=${{OUTPUT}}"
echo "ITER_INDEX=${{ITER_INDEX}}"
echo "ITER_START=${{ITER_START}}"
echo "ITER_END=${{ITER_END}}"

cd "${{REPO_DIR}}"

ENV_DIR="${{ENV_DIR}}" \\
OUTPUT_DIR="${{OUTPUT_DIR}}" \\
OUTPUT="${{OUTPUT}}" \\
ITER_INDEX="${{ITER_INDEX}}" \\
ITER_START="${{ITER_START}}" \\
ITER_END="${{ITER_END}}" \\
READER_WORKERS="${{READER_WORKERS}}" \\
COMPRESS_WORKERS="${{COMPRESS_WORKERS}}" \\
COMPRESS_CHUNK_RECORDS="${{COMPRESS_CHUNK_RECORDS}}" \\
GZIP_COMPRESSLEVEL="${{GZIP_COMPRESSLEVEL}}" \\
bash scripts/corpus/training-batch-replay/replay_training_task.sh "${{TASK_DIR}}"

chmod -R g+rwX "${{OUTPUT_DIR}}" || true
echo "Done. Output: ${{OUTPUT}}"
"""


def shell_quote(value: object) -> str:
    return shlex.quote(str(value))


def shell_quote_optional(value: object | None) -> str:
    if value is None:
        return "''"
    return shell_quote(value)


def default_repo_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def format_range_label(iter_index: int | None, iter_start: int | None, iter_end: int | None) -> str:
    if iter_index is not None:
        return f"iter_{iter_index:07d}"
    if iter_start is None and iter_end is None:
        return "global-batches"
    start = 0 if iter_start is None else iter_start
    end = "end" if iter_end is None else f"{iter_end:07d}"
    return f"iter_{start:07d}-{end}"


def format_output_path(output_dir: str, range_label: str) -> Path:
    return Path(output_dir) / f"{range_label}.jsonl.gz"


def load_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and submit an ABCI qsub script for training-batch replay.")
    parser.add_argument("--task-dir", default=DEFAULT_TASK_DIR, help="Absolute path to the source training task directory.")
    parser.add_argument("--env-dir", default=DEFAULT_ENV_DIR, help="Absolute path to the source training environment directory.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Absolute output directory for replayed batches.")
    parser.add_argument(
        "--iter-index",
        type=int,
        default=None,
        help=(
            "Replay only this training iteration/global batch index. This is "
            "a shorthand for --iter-start INDEX --iter-end INDEX+1."
        ),
    )
    parser.add_argument(
        "--iter-start",
        type=int,
        default=None,
        help="First training iteration/global batch index to replay, inclusive.",
    )
    parser.add_argument(
        "--iter-end",
        type=int,
        default=None,
        help=(
            "Training iteration end, exclusive. Defaults to the end of the "
            "training loop when --iter-start is set."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Absolute output JSONL gzip path. Defaults to a name derived "
            "from --iter-index, --iter-start, and --iter-end."
        ),
    )
    parser.add_argument("--pbs-job-name", required=True, help="Name of the PBS job.")
    parser.add_argument("--overwrite", action="store_true", help="Allow submitting even when the output file already exists.")

    parser.add_argument("--repo-dir", default=str(default_repo_dir()), help=argparse.SUPPRESS)
    parser.add_argument("--reader-workers", type=int, default=64, help=argparse.SUPPRESS)
    parser.add_argument("--compress-workers", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--compress-chunk-records", type=int, default=512, help=argparse.SUPPRESS)
    parser.add_argument("--gzip-compresslevel", type=int, default=0, help=argparse.SUPPRESS)

    parser.add_argument("--pbs-queue", default="R9920261000", help=argparse.SUPPRESS)
    parser.add_argument("--pbs-group", default="gcg51557", help=argparse.SUPPRESS)
    parser.add_argument("--resource-type", default="rt_HF", help=argparse.SUPPRESS)
    parser.add_argument("--select", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--ncpus", type=int, default=192, help=argparse.SUPPRESS)
    parser.add_argument("--walltime", default="24:00:00", help=argparse.SUPPRESS)
    parser.add_argument("--pbs-option", dest="pbs_options", nargs="*", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="Print the generated qsub script without submitting.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> tuple[str, Path]:
    if args.iter_index is not None and (args.iter_start is not None or args.iter_end is not None):
        raise ValueError("--iter-index cannot be used with --iter-start or --iter-end.")
    if args.iter_index is not None and args.iter_index < 0:
        raise ValueError("--iter-index must be non-negative.")
    if args.iter_start is not None and args.iter_start < 0:
        raise ValueError("--iter-start must be non-negative.")
    if args.iter_end is not None and args.iter_end < 0:
        raise ValueError("--iter-end must be non-negative.")
    if args.iter_start is not None and args.iter_end is not None and args.iter_end < args.iter_start:
        raise ValueError("--iter-end must be greater than or equal to --iter-start.")

    for attr in ["task_dir", "env_dir", "output_dir", "repo_dir"]:
        value = getattr(args, attr)
        if not os.path.isabs(value):
            raise ValueError(f"{attr.replace('_', '-')} must be an absolute path: {value}")

    task_dir = Path(args.task_dir)
    if not task_dir.is_dir():
        raise ValueError(f"Task directory does not exist: {task_dir}")

    env_dir = Path(args.env_dir)
    if not env_dir.is_dir():
        raise ValueError(f"Environment directory does not exist: {env_dir}")

    replay_task = Path(args.repo_dir) / "scripts" / "corpus" / "training-batch-replay" / "replay_training_task.sh"
    if not replay_task.is_file():
        raise ValueError(f"Replay task script does not exist: {replay_task}")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path exists and is not a directory: {output_dir}")

    range_label = format_range_label(args.iter_index, args.iter_start, args.iter_end)
    output = Path(args.output) if args.output else format_output_path(args.output_dir, range_label)
    if not output.is_absolute():
        raise ValueError(f"output must be an absolute path: {output}")
    if output.exists() and not args.overwrite:
        raise ValueError(f"Output file already exists: {output}. Use --overwrite to submit anyway.")

    for attr in ["reader_workers", "compress_workers", "compress_chunk_records", "select", "ncpus"]:
        if getattr(args, attr) <= 0:
            raise ValueError(f"--{attr.replace('_', '-')} must be positive.")
    if not 0 <= args.gzip_compresslevel <= 9:
        raise ValueError("--gzip-compresslevel must be between 0 and 9.")

    return range_label, output


def main() -> None:
    args = load_args()
    range_label, output = check_args(args)
    output_dir = Path(args.output_dir)

    qsub_script = TEMPLATE.format(
        job_name=args.pbs_job_name,
        pbs_group=args.pbs_group,
        pbs_queue=args.pbs_queue,
        resource_type=args.resource_type,
        select=args.select,
        ncpus=args.ncpus,
        walltime=args.walltime,
        output_dir=str(output_dir),
        range_label=range_label,
        pbs_options="\n".join(args.pbs_options),
        repo_dir=shell_quote(args.repo_dir),
        env_dir=shell_quote(args.env_dir),
        task_dir=shell_quote(args.task_dir),
        output=shell_quote(output),
        iter_index=shell_quote_optional(args.iter_index),
        iter_start=shell_quote_optional(args.iter_start),
        iter_end=shell_quote_optional(args.iter_end),
        reader_workers=shell_quote(args.reader_workers),
        compress_workers=shell_quote(args.compress_workers),
        compress_chunk_records=shell_quote(args.compress_chunk_records),
        gzip_compresslevel=shell_quote(args.gzip_compresslevel),
    )

    if args.dry_run:
        print(qsub_script)
        return

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(output_dir, 0o775)
    os.chmod(logs_dir, 0o775)

    qsub_script_path = logs_dir / f"qsub_{range_label}.sh"
    qsub_script_path.write_text(qsub_script)
    os.chmod(qsub_script_path, 0o664)

    config = dict(args._get_kwargs())
    config["range_label"] = range_label
    config["output"] = str(output)
    config_path = logs_dir / f"qsub_{range_label}.json"
    config_path.write_text(json.dumps(config, indent=4, ensure_ascii=False))
    os.chmod(config_path, 0o664)

    result = subprocess.run(
        ["qsub", str(qsub_script_path)],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    job_id = result.stdout.strip()
    logging.info("JOB ID: %s", job_id)
    print(job_id)


if __name__ == "__main__":
    main()
