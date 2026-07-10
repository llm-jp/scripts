"""Remove low-quality documents."""

import argparse
import logging
import json
import os
import concurrent
import concurrent.futures
from typing import Iterator, TextIO

import tqdm

logger = logging.getLogger(__name__)


def get_line_break_or_white_space_ratio(text: str) -> float:
    """Get the ratio of line breaks and white spaces in the text.

    Args:
        text: The text to analyze.

    Returns:
        The ratio of line breaks and white spaces in the text.
    """
    if text == "":
        return 0.0
    return (text.count("\n") + text.count(" ")) / len(text)


def get_short_line_ratio(text: str, threshold: int = 5) -> float:
    """Get the ratio of characters in short lines in the text.

    Args:
        text: The text to analyze.
        threshold: The threshold to determine if a line is short.

    Returns:
        The ratio of short lines in the text.
    """
    if text == "":
        return 0.0
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) == 0:
        return 0.0
    short_lines = [line for line in lines if len(line.replace(" ", "")) <= threshold]
    return sum(map(len, short_lines)) / sum(map(len, lines))


def process_line(
    line: str,
    line_break_or_white_space_ratio_threshold: float = 0.2,
    short_line_ratio_threshold: float = 0.1,
) -> str:
    """Process a line in the input file.

    Args:
        line: A line in the input file.
        line_break_or_white_space_ratio_threshold: The threshold of the ratio of line breaks and white spaces.
        short_line_ratio_threshold: The threshold of the ratio of short lines.
    """
    try:
        row = json.loads(line)
    except Exception as e:
        logging.error(f"Error: {e}")
        return line
    text = row["text"]
    orig_meta = row.get("meta", {})
    row["meta"] = {
        "line_break_or_white_space_ratio": get_line_break_or_white_space_ratio(text),
        "short_line_ratio": get_short_line_ratio(text),
    }
    if orig_meta:
        row["meta"]["meta"] = orig_meta
    if (
        row["meta"]["line_break_or_white_space_ratio"]
        > line_break_or_white_space_ratio_threshold
    ):
        return ""
    if row["meta"]["short_line_ratio"] > short_line_ratio_threshold:
        return ""
    return json.dumps(row, ensure_ascii=False) + "\n"


def process_lines(lines: list[str]) -> str:
    """Process lines.

    Args:
        lines (list[str]): Input lines.

    Returns:
        str: Processed lines.
    """
    ret: str = ""
    for line in lines:
        ret += process_line(line)
    return ret


def buffered_read(file: TextIO, buffer_size: int = 32) -> Iterator[list[str]]:
    """Buffered read.

    Args:
        file: File object.
        processed_ids: Processed IDs.
        buffer_size: Buffer size.

    Yields:
        str: Line.
    """
    lines: list[str] = []
    for line in file:
        lines.append(line)
        if len(lines) == buffer_size:
            yield lines
            lines = []
    yield lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the output file.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers for multiprocessing.",
    )
    parser.add_argument("--buffer-size", type=int, default=256, help="Buffer size.")
    args = parser.parse_args()

    num_workers = args.num_workers if args.num_workers != -1 else os.cpu_count()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with (
        open(args.input_file, "rt", encoding="utf-8") as fin,
        open(args.output_file, "wt", encoding="utf-8") as fout,
    ):
        with concurrent.futures.ProcessPoolExecutor(num_workers) as executor:
            futures = []
            for lines in buffered_read(fin, buffer_size=args.buffer_size):
                futures.append(executor.submit(process_lines, lines))
            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
            ):
                fout.write(future.result())
                fout.flush()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
