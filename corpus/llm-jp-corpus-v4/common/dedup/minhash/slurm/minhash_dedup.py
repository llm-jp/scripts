from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (MinhashConfig,
                                              MinhashDedupBuckets,
                                              MinhashDedupCluster,
                                              MinhashDedupFilter)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import Languages

WORK_DIR = "/home/shared/experiments/0118_dedup_corpusv4_ja"
VENV_PATH = f"{WORK_DIR}/environment/.venv/bin/activate"


@dataclass
class Args:
    input: str
    output: str
    ngram: int
    buckets: int
    hashes_per_bucket: int
    venv: str


argparser = ArgumentParser()
argparser.add_argument("input", type=str)
argparser.add_argument("output", type=str)
argparser.add_argument("--ngram", default=5, type=int)
argparser.add_argument("-r", "--buckets", default=20, type=int)
argparser.add_argument("-b", "--hashes_per_bucket", default=10, type=int)
argparser.add_argument("--venv", default=VENV_PATH, type=str)
args = argparser.parse_args(namespace=Args)


MINHASH_DIRNAME = (
    f"minhash-{args.ngram}gram-{args.buckets}buckets-{args.hashes_per_bucket}hashes"
)
MINHASH_DIR = Path(args.output) / MINHASH_DIRNAME
RESULT_DIR = f"{MINHASH_DIR}/results"
LOG_DIR = f"{MINHASH_DIR}/logs"
SLURM_LOGS_DIR = f"{MINHASH_DIR}/slurm_logs"

all_files = [_f for _f in Path(args.input).rglob("*") if _f.resolve().is_file()]
TOTAL_TASKS = len(all_files)

# this is the original data that we want to deduplicate
INPUT_READER = JsonlReader(args.input)
# you can also change ngrams or the number of buckets and their size here
minhash_config = MinhashConfig(
    hash_config=HashConfig(precision=64),
    n_grams=args.ngram,
    num_buckets=args.buckets,
    hashes_per_bucket=args.hashes_per_bucket,
)  # better precision -> fewer false positives (collisions)


# stage 1 computes minhash signatures for each task (each task gets a set of files)
stage1 = SlurmPipelineExecutor(
    job_name="calc_minhash",
    pipeline=[
        INPUT_READER,
        MinhashDedupSignature(
            output_folder=f"{RESULT_DIR}/signatures",
            config=minhash_config,
            language=Languages.japanese,
            skip_existing_sigs=True,
        ),
    ],
    tasks=TOTAL_TASKS,
    time="48:00:00",
    partition="cpu",
    logging_dir=f"{LOG_DIR}/signatures",
    slurm_logs_folder=SLURM_LOGS_DIR,
    venv_path=args.venv,
    max_array_launch_parallel=True,
    stagger_max_array_jobs=5,
)

# stage 2 finds matches between signatures in each bucket
stage2 = SlurmPipelineExecutor(
    job_name="match_minhash",
    pipeline=[
        MinhashDedupBuckets(
            input_folder=f"{RESULT_DIR}/signatures",
            output_folder=f"{RESULT_DIR}/buckets",
            config=minhash_config,
        ),
    ],
    tasks=minhash_config.num_buckets,
    time="120:00:00",
    partition="cpu",
    logging_dir=f"{LOG_DIR}/buckets",
    depends=stage1,
    slurm_logs_folder=SLURM_LOGS_DIR,
    venv_path=args.venv,
    max_array_launch_parallel=True,
    stagger_max_array_jobs=5,
)

# stage 3 creates clusters of duplicates using the results from all buckets
stage3 = SlurmPipelineExecutor(
    job_name="calc_cluster",
    pipeline=[
        MinhashDedupCluster(
            input_folder=f"{RESULT_DIR}/buckets",
            output_folder=f"{RESULT_DIR}/remove_ids",
            config=minhash_config,
            save_cluster_id=True,
            save_cluster_size=True,
        ),
    ],
    tasks=1,
    time="120:00:00",
    partition="cpu",
    logging_dir=f"{LOG_DIR}/clusters",
    mem_per_cpu_gb=70,
    cpus_per_task=2,
    depends=stage2,
    slurm_logs_folder=SLURM_LOGS_DIR,
    venv_path=args.venv,
    max_array_launch_parallel=True,
    stagger_max_array_jobs=5,
)

# stage 4 reads the original input data and removes all but 1 sample per duplicate cluster
# the data must match exactly stage 1, so number of tasks and the input source must be the same
stage4 = SlurmPipelineExecutor(
    job_name="remove_duplicate",
    pipeline=[
        INPUT_READER,
        TokensCounter(),  # nice way to see how many tokens we had before and after deduplication
        MinhashDedupFilter(
            input_folder=f"{RESULT_DIR}/remove_ids",
            exclusion_writer=JsonlWriter(f"{RESULT_DIR}/removed"),
            load_cluster_ids=True,
            load_cluster_sizes=True,
        ),
        JsonlWriter(output_folder=f"{RESULT_DIR}/deduplicated_output"),
    ],
    tasks=TOTAL_TASKS,
    time="120:00:00",
    partition="cpu",
    logging_dir=f"{LOG_DIR}/filter",
    depends=stage3,
    slurm_logs_folder=SLURM_LOGS_DIR,
    venv_path=args.venv,
    max_array_launch_parallel=True,
    stagger_max_array_jobs=5,
)


stage4.run()
