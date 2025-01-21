"""
This script processes fineweb-edu-score-2 corpus, splits the data based on a int(score * n), and saves the results into .jsonl.gz files.

Usage:
    python src/main.py \
        --fineweb-edu-dir <input_directory> \
        --output-dir <output_directory> \
        --n <split_factor> \
        --num-workers <number_of_workers> \
        --cache-size <cache_size>
"""

import json
import gzip
import shutil
import argparse

from pathlib import Path
from multiprocessing import cpu_count
from multiprocessing import Pool

from collections import defaultdict

from tqdm import tqdm

from datatrove.pipeline.readers import ParquetReader


def load_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fineweb-edu-dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Split file by int(score * n)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=cpu_count(),
        help="Number of workers",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=3000,
        help="Cache size per int(score * n)",
    )
    return parser.parse_args()


def split_by_score(file_dir, output_dir, n, cache_size):
    # Get the sub-directory name (e.g. CC-MAIN-2013-20)
    *_, sub_dir = file_dir.parts

    # Create flag directory and path
    flag_dir = Path(output_dir) / "flag"
    flag_dir.mkdir(parents=True, exist_ok=True)
    flag_path = flag_dir / (sub_dir + ".flag")

    # Check if the flag file exists
    if flag_path.exists():
        return

    # Create output directory
    output_dir = Path(output_dir) / sub_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize ParquetReader to .parquet files
    data_reader = ParquetReader(str(file_dir), file_progress=True)

    # Initialize cache
    cache = defaultdict(list)
    for document in data_reader():
        # Calculate rounded score
        int_round_score = int(document.metadata["score"] * n)

        # Prepare document data
        d = {
            "id": document.id,
            "text": document.text,
            "media": document.media,
            "metadata": document.metadata,
        }

        # Add document to cache
        cache[int_round_score].append(d)

        # Write cache to file if cache size is reached
        if len(cache[int_round_score]) >= cache_size:
            dumps = list(map(json.dumps, cache[int_round_score]))
            output_path = output_dir / f"{int_round_score}.jsonl.gz"
            with gzip.open(output_path, "at") as f:
                f.write("\n".join(dumps) + "\n")
            cache[int_round_score] = []

    # Write remaining documents in cache to file
    for score, docs in cache.items():
        if not docs:
            continue

        output_path = output_dir / f"{score}.jsonl.gz"
        dumps = list(map(json.dumps, docs))
        with gzip.open(output_path, "at") as f:
            f.write("\n".join(dumps) + "\n")

    # Create flag file
    flag_path.touch()


def wrp_split_by_score(args):
    # Wrapper function for split_by_score
    split_by_score(*args)


def main():
    # Main function to load arguments and start processing
    args = load_args()

    # Get list of file sub directories (e.g. CC-MAIN-2013-20)
    file_dirs = Path(args.fineweb_edu_dir).glob("*")

    # Prepare jobs for multiprocessing
    jobs = [
        (file_dir, args.output_dir, args.n, args.cache_size) for file_dir in file_dirs
    ]
    # Use multiprocessing pool to process files
    with Pool(args.num_workers) as p, tqdm(total=len(jobs)) as pbar:
        for _ in p.imap_unordered(wrp_split_by_score, jobs):
            pbar.update()


if __name__ == "__main__":
    main()
