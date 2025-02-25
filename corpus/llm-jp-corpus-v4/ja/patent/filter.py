"""
This script reads JSONL files from a specified input directory, extracts text matching
predefined patterns, and writes the extracted text to new JSONL files in an output directory.
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x: x


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

EXTRACT_FEATURES = {
    "A": [["(57)【要約】"]],
    "B9": [["(57)【特許請求の範囲】"]],
    "T": [["(57)【要約】"]],
    "S": [["(57)【要約】"]],
    "A5": [["【手続補正書】", "【誤訳訂正書】"]],
    "APC": [["【事件の表示】"], ["【審決日】", "【審理終結日】"]],
    "T5": [["【手続補正書】", "【誤訳訂正書】"]],
    "U9": [["(57)【要約】"]],
    "APD": [
        ["【事件の表示】", "【訂正の要旨】"],
        ["【審判長】", "（２１）【出願番号】"],
    ],
    "ATC": [["【事件の表示】"], ["【審決日】", "【審理終結日】"]],
}


def is_match_patterns(text: str, patterns: list[str]) -> bool:
    """
    Check if any of the given patterns is found in the provided text.

    Args:
        text (str): The text to be searched.
        patterns (list[str]): A list of pattern strings to search for in the text.

    Returns:
        bool: True if at least one pattern is found in the text; otherwise, False.
    """
    return any(_pattern in text for _pattern in patterns)


def extract_text_from_file(input_file: Path, output_dir: Path) -> None:
    """
    Read a single JSONL file, extract text according to predefined patterns,
    and write the extracted text to a new JSONL file.

    Extraction is performed by identifying start and end patterns defined
    in the EXTRACT_FEATURES dictionary.

    Args:
        input_file (Path): Path to the input JSONL file.
        output_dir (Path): Path to the directory where the output file will be written.

    Raises:
        AssertionError: If no text matches the defined extraction patterns.
    """
    symbol = input_file.stem[5:]
    extract_range = EXTRACT_FEATURES[symbol]
    output_file = output_dir / input_file.name

    with input_file.open() as f_read, output_file.open("w", buffering=10**7) as f_write:
        for i, line in enumerate(f_read):
            data: dict[str, list[str]] = json.loads(line)
            write_dict = {k: v for k, v in data.items() if k != "text"}
            write_text = []

            do_write = False
            for _inner_line in data["text"].split("\n"):
                # Start extraction if the line matches any start pattern
                if not do_write and is_match_patterns(_inner_line, extract_range[0]):
                    do_write = True

                if do_write:
                    # If a second pattern list exists and matches, stop extraction
                    if len(extract_range) == 2 and is_match_patterns(
                        _inner_line, extract_range[1]
                    ):
                        do_write = False
                        continue
                    write_text.append(_inner_line)

            # Ensure we actually extracted something
            assert (
                len(write_text) > 0
            ), "No matching text found in document.\n\tFile: {}\n\tLine: {}\n\tFeature: {}".format(
                input_file, i, extract_range[0]
            )

            write_dict["text"] = "\n".join(write_text)
            f_write.write(json.dumps(write_dict, ensure_ascii=False) + "\n")


def main(input_dir: Path, output_dir: Path, worker: int = 16) -> None:
    """
    Traverse the input directory to find JSONL files matching known patterns,
    extract relevant text from each file, and write it to the output directory.

    Args:
        input_dir (Path): Path to the directory containing the input JSONL files.
        output_dir (Path): Path to the directory where extracted files will be created.
        worker (int, optional): Number of worker processes for parallel file handling. Defaults to 16.
    """
    logger.debug(f"{input_dir=}")
    logger.debug(f"{output_dir=}")
    all_files = [
        _f for _f in input_dir.glob("*.jsonl") if _f.stem[5:] in EXTRACT_FEATURES
    ]
    extract_text_with_outdir = partial(extract_text_from_file, output_dir=output_dir)

    with ProcessPoolExecutor(max_workers=worker) as executor:
        futures = {
            executor.submit(extract_text_with_outdir, file): file for file in all_files
        }

        for future in tqdm(as_completed(futures), total=len(futures), ncols=0):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing {futures[future]}: {e}")


if __name__ == "__main__":

    class Args:
        """
        A placeholder class to store command-line arguments provided by argparse.
        """

        input: str
        output: str
        worker: int

    parser = argparse.ArgumentParser(
        description="Read JSONL files, extract text based on predefined patterns, and write the results to output."
    )
    parser.add_argument(
        "input", type=str, help="Path to the input directory containing JSONL files."
    )
    parser.add_argument(
        "output", type=str, help="Path to the output directory for extracted files."
    )
    parser.add_argument(
        "--worker",
        type=int,
        default=64,
        help="Number of parallel workers for processing (default: 64).",
    )
    args = parser.parse_args(namespace=Args)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    assert (
        input_dir.exists() and input_dir.is_dir()
    ), "Input directory does not exist or is not a directory."
    if not output_dir.exists():
        logger.info("Creating directory: %s", output_dir)
        output_dir.mkdir(parents=True)
    else:
        logger.error("Directory already exists: %s", output_dir)
        sys.exit()

    main(input_dir, output_dir, args.worker)
