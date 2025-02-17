"""
Generate text length statistics for files in given directory.
"""

import argparse
import json
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def flexible_open(file_path: Path):
    if file_path.suffix == ".gz":
        import gzip

        return gzip.open(file_path, "rb")
    else:
        try:
            return file_path.open()
        except Exception as e:
            logger.error("This extension is not supported: %s", file_path)
            raise e


def process_file(file: Path):
    f = flexible_open(file)
    return [len(json.loads(line)["text"]) for line in f]


def show_hist(data: list[int]):
    """
    Generates and saves a histogram of character counts and cumulative sums.

    Args:
        data (list[int]): List of character lengths.
    """

    counter = Counter(data)
    sorted_counter = np.array(sorted(counter.keys()))
    frequencies = np.array([counter[count] for count in sorted_counter])
    cumulative_sums = np.cumsum(sorted_counter * frequencies)

    plt.figure(figsize=(12, 5))
    plt.suptitle("文字数の分布と累積合計(重複削除前)", fontsize=16, fontweight="bold")
    plt.subplot(1, 2, 1)
    plt.bar(
        sorted_counter,
        frequencies,
        width=10,
        color="gray",
        edgecolor="black",
        alpha=0.7,
    )
    plt.xlabel("文字数")
    plt.ylabel("出現頻度")
    plt.title("文字数の分布")
    plt.xscale("log")
    plt.yscale("log")

    plt.subplot(1, 2, 2)
    plt.plot(sorted_counter, cumulative_sums, marker="o", linestyle="-", color="b")
    plt.xlabel("文字数")
    plt.ylabel("累積合計文字数")
    plt.title("累積合計文字数の推移")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # グラフを表示
    plt.tight_layout()
    plt.savefig("char_distribution.png", dpi=300, bbox_inches="tight")


def main(input_dir: Path, worker=32):
    lens = []
    all_files = [_f for _f in input_dir.rglob("*") if _f.is_file()]
    with ProcessPoolExecutor(max_workers=worker) as executor:
        futures = executor.map(process_file, all_files)
        logger.info("Processing files...")
        for _lens in tqdm(futures, total=len(all_files), ncols=0):
            lens.extend(_lens)

    show_hist(lens)


if __name__ == "__main__":

    class Args:
        dir: str
        worker: int

    argparser = argparse.ArgumentParser(
        description="Process JSONL files and generate text length distributions."
    )
    argparser.add_argument(
        "dir", type=str, help="Path to the input directory containing JSONL files."
    )
    argparser.add_argument(
        "--worker",
        type=int,
        default=16,
        help="Number of parallel workers for processing (default: 16).",
    )
    args = argparser.parse_args(namespace=Args)

    assert Path(args.dir).exists()
    main(Path(args.dir), args.worker)
