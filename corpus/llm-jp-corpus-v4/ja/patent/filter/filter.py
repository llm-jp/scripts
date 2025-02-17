"""
This script processes JSONL files by extracting and filtering structured text data based on predefined regex patterns.

It reads JSONL files from an input directory, applies text filtering based on character length, and stores the processed
output in structured subdirectories based on matched text patterns. The script supports parallel processing using
multiple worker threads.

Usage:
    python script.py <input_dir> <output_dir> --char-limit <min_length> --worker <num_workers>

Arguments:
    input_dir (str): Path to the input directory containing JSONL files.
    output_dir (str): Path to the output directory where processed files will be stored.
    --char-limit (int, optional): Minimum character length for filtering. Default is 0.
    --worker (int, optional): Number of parallel workers for processing. Default is 64.

Example:
    python script.py ./input ./output --char-limit 100 --worker 32

"""

import argparse
import copy
import json
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Iterator, TextIO

from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

extract_patterns = {
    "seikyuukou": r"請求項\d+",
    "tokkyo_koumoku": r"\d{4}",
    "jijitsu_and_riyuu": r"事実及び理由",
    "riyuu_1": r"理　由",
    "hatsumei_syousai": r"発明の詳細な説明",
    "kaiketsu": r"解決する",
    "kaiketsu": r"解決しよう",
    "juurai_gijutsu": r"従来の技術",
    "hatsumei_kouka": r"発明の効果",
    "deprecated-zumen_setsumei": r"図面の簡単な説",
    "deprecated-hugou_setsumei": r"符号の説明",
    "hatsumei_jissi_sairyou": r"発明を実施するための最良の形態",
    "deprecated-jissirei": r"実施例\d*",
    "riyuu_2": r"理由",
    "deprecated-national_pamphlet": r"国際公開パンフレット",
    "riyuu_3": r"理  由",
    "tokkyo_hanni": r"特許請求の範囲",
    "teisei_youshi": r"訂正の要旨",
    "jijitsu": r"事実",
    "souiten": r"相違点",
    "hatsumei_haikei": r"発明の背景",
    "hatsumei_jissi": r"発明の実施形態",
    "deprecated-isyou_tokutyou": r"意匠の特徴",
}

extract_patterns = {k: re.compile(v) for k, v in extract_patterns.items()}


def pattern_filter(key: str) -> str | None:
    """
    Filters a given key based on predefined regex patterns.

    Args:
        key (str): Input key to filter.

    Returns:
        str | None: Mapped key if a pattern matches, otherwise None.
    """
    for _save_to, _pattern in extract_patterns.items():
        if _pattern.search(key):
            return _save_to
    return None


def line_process_iterator(line: str, char_limit: int = 0) -> Iterator[tuple[str, dict]]:
    """
    Processes a JSON line and filters its contents based on text length.

    Args:
        line (str): JSON line to process.
        char_limit (int, optional): Minimum character limit. Defaults to 0.

    Yields:
        tuple[str, dict]: Mapped key and processed JSON dictionary.
    """
    data: dict[str, list[str]] = json.loads(line)
    meta = data["meta"]
    for key, documents in data.items():
        if key == "meta":
            continue
        _save_to = pattern_filter(key)
        if _save_to is None:
            continue
        for i, _doc in enumerate(documents):
            if len(_doc.strip()) <= char_limit:
                continue
            _meta = copy.deepcopy(meta)
            _meta["key"] = key
            _meta["order"] = i
            write_dict = {"text": _doc.strip(), "meta": _meta}
            yield _save_to, write_dict


def process_file(input_file: Path, output_dir: Path, char_limit: int = 0):
    """
    Processes a file and saves filtered results in structured output directories.

    Args:
        input_file (Path): Path to the input JSONL file.
        output_dir (Path): Directory where output files will be stored.
        char_limit (int, optional): Minimum character limit. Defaults to 0.
    """
    f = input_file.open()
    save_files: dict[str, TextIO] = {}
    for line in f:
        for _save_to, write_dict in line_process_iterator(line, char_limit=char_limit):
            output_file = output_dir / _save_to / input_file.name
            if not output_file.parent.exists():
                output_file.parent.mkdir()
            if _save_to not in save_files:
                save_files[_save_to] = output_file.open("w", buffering=10**5)
            save_files[_save_to].write(
                json.dumps(write_dict, ensure_ascii=False) + "\n"
            )

    # finalize
    f.close()
    for _file in save_files.values():
        _file.close()


def main(input_dir: Path, output_dir: Path, char_limit: int, worker=16):
    """
    Main function to process JSONL files in parallel.

    Args:
        input_dir (Path): Input directory containing JSONL files.
        output_dir (Path): Output directory for processed files.
        char_limit (int): Minimum character length for filtering.
        worker (int, optional): Number of parallel workers. Defaults to 16.
    """
    logger.debug(f"{input_dir=}")
    logger.debug(f"{output_dir=}")
    logger.debug(f"{char_limit=}")
    all_files = list(input_dir.glob("*.jsonl"))
    process_file_with_outdir = partial(
        process_file, output_dir=output_dir, char_limit=char_limit
    )
    with ProcessPoolExecutor(max_workers=worker) as executor:
        futures = {
            executor.submit(process_file_with_outdir, file): file for file in all_files
        }

        for future in tqdm(as_completed(futures), total=len(futures), ncols=80):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing {futures[future]}: {e}")


if __name__ == "__main__":

    class Args:
        input: str
        output: str
        char_limit: int
        worker: int

    parser = argparse.ArgumentParser(
        description="Process JSONL files and filter text data based on length."
    )
    parser.add_argument(
        "input", type=str, help="Path to the input directory containing JSONL files."
    )
    parser.add_argument(
        "output", type=str, help="Path to the output directory for processed files."
    )
    parser.add_argument(
        "-char",
        "--char-limit",
        type=int,
        default=0,
        help="Minimum character length for filtering (default: 0).",
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
    assert input_dir.exists() and input_dir.is_dir()
    if not output_dir.exists():
        logger.info("Create directory: %s", output_dir)
        output_dir.mkdir(parents=True)
    else:
        logger.error("Directory already exists: %s", output_dir)
        sys.exit()

    main(input_dir, output_dir, args.char_limit, args.worker)
