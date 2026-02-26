import gzip
import json
import multiprocessing
import sys
from pathlib import Path

from tqdm import tqdm

input_dir = "/model/experiments/0118_dedup_corpusv4_ja/data/all/minhash-5gram-20buckets-10hashes/results/deduplicated_output"
parallel_jobs = 32

# fineweb-2 preserve original path on original fineweb-2
patterns = ["s3://commoncrawl/crawl-data", "/fsx/guilherme/cc2023-50"]


def convert_patterns(path: str) -> str:
    """
    Normalize the file path based on known prefix patterns.

    Examples:
        >>> convert_patterns("s3://commoncrawl/crawl-data/CC-MAIN-2023/file1")
        "s3://commoncrawl/crawl-data"

        >>> convert_patterns("/data/local/custom_corpus/file3")
        "/data/local/custom_corpus/file3"
    """
    for _pat in patterns:
        if _pat in path:
            return _pat
    return path


def process_file(file):
    unique_paths = set()

    try:
        with gzip.open(file,"rt") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    file_path = data.get("metadata").get("file_path")
                    converted_path = convert_patterns(file_path)
                    unique_paths.add(converted_path)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error processing {file}: {e}", file=sys.stderr)
        return None

    if len(unique_paths) != 1:
        print(f"Warning: {file} has {len(unique_paths)} unique values!")


def main():
    files = list(Path(input_dir).rglob("*.jsonl.gz"))

    with multiprocessing.Pool(parallel_jobs) as pool:
        list(
            tqdm(
                pool.imap_unordered(process_file, files),
                total=len(files),
                desc="Processing files",
                ncols=0,
            ),
        )


if __name__ == "__main__":
    main()
