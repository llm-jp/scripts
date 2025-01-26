"""
This script filters out duplicate documents from the fineweb dataset based on the fineweb-edu dataset.

Usage:
    python filter.py \
        --fineweb-edu-dir <input_directory_for_fineweb_edu> \
        --fineweb-dir <input_directory_for_fineweb> \
        --output-dir <output_directory> \
        --num-workers <number_of_workers> \
        --cache-size <cache_size>
"""

import json
import gzip
import shutil
import logging
import argparse

from pathlib import Path
from multiprocessing import cpu_count
from multiprocessing import Pool

from tqdm import tqdm

from datatrove.pipeline.readers import ParquetReader

import pyarrow.parquet as pq


def load_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fineweb-edu-dir",
        type=str,
        default="/model/llm-jp-corpus/v4.0.0/download/fineweb-edu-score-2/data",
    )
    parser.add_argument(
        "--fineweb-dir",
        type=str,
        default="/model/llm-jp-corpus/v4.0.0/download/fineweb/data",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/model/llm-jp-corpus/v4.0.0/sample/en_fineweb",
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
        help="Cache size",
    )
    parser.add_argument(
        "--part",
        type=int,
        default=1,
        help="Which part to process (1-based index)",
    )
    parser.add_argument(
        "--total-parts",
        type=int,
        default=1,
        help="Total number of parts to divide the work into",
    )

    return parser.parse_args()


def get_all_ids(file_dir):
    parquet_paths = Path(file_dir).rglob("*.parquet")
    ids = []
    for path in tqdm(parquet_paths):
        table = pq.read_table(path, columns=["id"])
        ids += table.to_pandas().id.tolist()
    return set(ids)


def duplication_filter(file_dir, fineweb_edu_dir, output_dir, cache_size=10000):
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

    # Get all IDs from fineweb-edu data
    fineweb_edu_dir = Path(fineweb_edu_dir) / sub_dir
    fineweb_edu_ids = get_all_ids(fineweb_edu_dir)

    # Get list of .parquet files in the directory
    file_names = sorted(f.name for f in Path(file_dir).glob("*.parquet"))

    for file_name in file_names:
        # Initialize ParquetReader to read file_name.parquet files
        data_reader = ParquetReader(str(file_dir), glob_pattern=file_name)

        output_path = output_dir / file_name.replace(".parquet", ".jsonl.gz")

        # Initialize cache
        cache = []
        for document in data_reader():
            # Check for duplication
            if document.id in fineweb_edu_ids:
                fineweb_edu_ids.remove(document.id)  # Remove the ID from the set
                continue

            # Prepare document data
            d = {
                "id": document.id,
                "text": document.text,
                "media": document.media,
                "metadata": document.metadata,
            }
            cache.append(d)

            # Write cache to file if cache size is reached
            if len(cache) >= cache_size:
                dumps = list(map(json.dumps, cache))
                with gzip.open(output_path, "at") as f:
                    f.write("\n".join(dumps) + "\n")
                cache = []

        # Write remaining documents in cache to file
        if cache:
            dumps = list(map(json.dumps, cache))
            with gzip.open(output_path, "at") as f:
                f.write("\n".join(dumps) + "\n")

    if len(fineweb_edu_ids) > 0:
        # Generator is not exhausted.
        # This means that there are some IDs that failed to filter.
        logging.warning(f"Failed to filter all documents in {sub_dir}")
        return

    # Create flag file
    flag_path.touch()


def wrp_duplication_filter(args):
    # Wrapper function for duplication_filter
    return duplication_filter(*args)


def main():
    # Main function to load arguments and start processing
    args = load_args()

    # Get list of file sub directories (e.g. CC-MAIN-2013-20)
    file_dirs = sorted(Path(args.fineweb_dir).glob("*"))

    # Calculate the range of files to process
    total_files = len(file_dirs)
    part_size = total_files // args.total_parts
    start_index = (args.part - 1) * part_size
    end_index = start_index + part_size if args.part < args.total_parts else total_files

    file_dirs = file_dirs[start_index:end_index]

    # Prepare jobs for multiprocessing
    jobs = [
        (file_dir, args.fineweb_edu_dir, args.output_dir, args.cache_size)
        for file_dir in file_dirs
    ]
    # Use multiprocessing pool to process files
    with Pool(args.num_workers) as p, tqdm(total=len(jobs)) as pbar:
        for _ in p.imap_unordered(wrp_duplication_filter, jobs):
            pbar.update()


if __name__ == "__main__":
    main()
