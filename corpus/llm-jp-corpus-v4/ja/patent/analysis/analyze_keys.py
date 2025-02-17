"""
Process files in a directory and compute statistics on text sizes.
"""

import argparse
import json
import logging
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator

from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def most_common_keys(key2lens: dict[str, list[int]]) -> Iterator[str]:
    """
    Sorts and yields keys based on the sum of their corresponding values in descending order.

    Args:
        key2lens (dict[str, list[int]]): Dictionary mapping keys to lists of integer values.

    Yields:
        str: Keys sorted by the total sum of their associated values.
    """
    logger.info("Sorting data...")
    key2alllen = {k: sum(v) for k, v in tqdm(key2lens.items(), ncols=0)}
    for key, _ in sorted(key2alllen.items(), key=lambda item: item[1], reverse=True):
        yield key


def merge_key2lens(key2lens_1: dict[str, list[int]], key2lens_2: dict[str, list[int]]):
    for k, v in key2lens_2.items():
        key2lens_1[k].extend(v)
    return key2lens_1


def process_file(file: Path):
    """
    Processes a JSONL file, extracting key-value pairs and calculating value lengths.

    Args:
        file (Path): Path to the input JSONL file.

    Returns:
        dict[str, list[int]]: Dictionary mapping keys to lists of integer lengths.
    """
    f = file.open()
    key2lens = defaultdict(list)
    for line in f:
        data = json.loads(line)
        new_key2lens = {
            k: [len(_v) for _v in v] for k, v in data.items() if k != "meta"
        }
        key2lens = merge_key2lens(key2lens, new_key2lens)
    return key2lens


def main(input_dir: Path, top: int | None = None, limit: int | None = None, worker=16):
    key2lens = defaultdict(list)
    all_files = [_f for _f in input_dir.rglob("*") if _f.is_file()]
    with ProcessPoolExecutor(max_workers=worker) as executor:
        futures = executor.map(process_file, all_files)
        logger.info("Processing files...")
        for new_key2lens in tqdm(futures, total=len(all_files), ncols=0):
            key2lens = merge_key2lens(key2lens, new_key2lens)

    logger.info(f"Show text size from larger ones")
    for i, key in enumerate(most_common_keys(key2lens)):
        if top and i >= top:
            break
        values = key2lens[key]
        mean = statistics.mean(values)
        if limit and mean < limit:
            break
        stdev = statistics.mean(values)
        count = len(values)
        print(f"key={key[:20]:<20}, {count=:>10} {mean=:>20,.2f}, {stdev=:>20,.2f}")


if __name__ == "__main__":

    class Args:
        dir: str
        top: int
        limit: float
        worker: int

    argparser = argparse.ArgumentParser(
        description="Process JSONL files and compute text statistics."
    )
    argparser.add_argument(
        "dir", type=str, help="Path to the input directory containing JSONL files."
    )
    argparser.add_argument(
        "--top", type=int, help="Number of top keys to display based on total length."
    )
    argparser.add_argument(
        "--limit", type=float, help="Minimum mean text size to display."
    )
    argparser.add_argument(
        "--worker",
        type=int,
        default=16,
        help="Number of parallel workers for processing (default: 16).",
    )
    args = argparser.parse_args(namespace=Args)
    args = argparser.parse_args(namespace=Args)

    assert Path(args.dir).exists()
    main(Path(args.dir), args.top, args.limit, args.worker)
