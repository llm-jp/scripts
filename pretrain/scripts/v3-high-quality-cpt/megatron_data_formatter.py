"""
This module processes dataset information from CSV files, adjusts token counts, 
and filters entries based on a YAML configuration file. It loads dataset paths, 
calculates total token sizes, and checks for duplicates, with optional detailed output.

CSV File Specification:
    The CSV file should have the following structure:
    - Each row represents a dataset entry.
    - The columns in the CSV file are:
        1. Unused (can be ignored by the script).
        2. Dataset path (a string representing the path to the dataset).
        3. Token size (an integer representing the number of tokens in the dataset).

    Example of a valid CSV format:
        , /path/to/dataset1.jsonl, 1000
        , /path/to/dataset2.jsonl, 1500
        , /path/to/dataset3.jsonl, 2000

YAML Configuration Specification:
    The YAML configuration file defines the datasets and optionally a common 
    base directory for shared file paths. Each dataset includes a path to the CSV file, 
    a repeat factor, and an optional filter for specific dataset patterns.

    Fields:
        - common: (optional) Defines shared directories or common settings.
        - datasets: Defines the datasets to be processed. Each dataset must include:
            * basedir: (optional) A Directory that can be referenced in each dataset
            to avoid repeating the full path.
            * file: Path to the CSV file containing dataset information. If a base 
              directory is provided in the common section, the `file` path will be 
              relative to this `basedir`.
            * repeat: A multiplier applied to the token sizes from the CSV file.
            * filter: (optional) A list of strings used to filter dataset paths.

    Example of a valid YAML configuration:

        common:
            corpus_root: /path/to/corpus_root
        
        datasets:
            ja:
                basedir: corpus_root
                file: ja.csv
                repeat: 1.0
                filter:
                  - "wiki"

Output:
    - Environment variables (such as TOTAL_TOKEN_SIZE and TRAIN_DATA_PATH) will be 
      printed to stdout.
    - Statistical information such as token size summaries and warnings will be 
      logged to stderr.                  

Usage:
    python3 megatron_data_formatter.py {yaml_config_file} [-f]

Arguments:
    yaml_config_file    Path to the YAML configuration file.
    -f, -force          Ignore duplicate dataset entries.
"""

import csv
import logging
from argparse import ArgumentParser
from pathlib import Path

import yaml

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Argument parser setup
argparser = ArgumentParser()
argparser.add_argument("yaml_config")
argparser.add_argument(
    "-f",
    "--force",
    action="store_true",
    help="Ignore duplicate dataset entries",
)


def process_info(
    info_file: Path, repeat: float, filters: list[str] | None = None
) -> list[str]:
    """
    Processes dataset info from the CSV file, applies the repeat factor to the token count,
    and optionally filters based on the provided patterns.

    Args:
        info_file (Path): Path to the CSV file containing dataset information.
        repeat (float): Multiplier for token sizes.
        filters (list[str], optional): List of filter patterns.

    Returns:
        list[str]: List of formatted dataset strings (token size and data path).
    """
    data_paths = []
    with open(info_file, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            token_count = round(int(row[2]) * repeat)
            data_path = row[1]
            # Filter by filters if provided, otherwise add the path
            if filters is None or any(pattern in data_path for pattern in filters):
                data_paths.append(f"{token_count} {data_path}")
    return data_paths


def check_load_dataset(train_data_path: str, force: bool = False) -> None:
    """
    Checks and displays the dataset details including token sizes and file names.
    Optionally detects and handles duplicate entries.

    Args:
        train_data_path (str): The TRAIN_DATA_PATH string containing dataset paths and token sizes.
        force (bool, optional): If True, ignore duplicates and continue. Defaults to False.
    """
    total_token_size = 0
    file_checker = set()

    lang_total_tokens: dict[str, int] = {}
    logger.info("%-5s %-20s %15s", "Lang", "File Name", "Token Size")

    data_entries = train_data_path.split()
    for i in range(0, len(data_entries), 2):
        token_size = int(data_entries[i])
        data_path = Path(data_entries[i + 1])
        lang = data_path.parent.name
        file_name = data_path.stem
        combination = f"{lang}/{file_name}"

        if combination in file_checker:
            common_msg = f"Duplicate entry for {combination}."
            if not force:
                raise ValueError(common_msg)
            else:
                logger.warning("Warning: %s", common_msg)
        else:
            file_checker.add(combination)

        logger.info("%-5s %-20s %15s", lang, file_name, f"{token_size:,}")
        lang_total_tokens[lang] = lang_total_tokens.get(lang, 0) + token_size

        total_token_size += token_size

    logger.info("\nSummary:")
    for lang, total in lang_total_tokens.items():
        logger.info("%-5s %15s", lang, f"{total:,}")
    logger.info("%-5s %15s", "ALL", f"{total_token_size:,}")

    print(f"export TOTAL_TOKEN_SIZE={total_token_size}")


def main(config_path: str, force: bool = False) -> None:
    """
    Main function to process datasets as defined in the configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.
        force (bool, optional): If True, ignore duplicates. Defaults to False.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Process each dataset from the config
    token_and_path: list[str] = []
    for params in config["datasets"].values():
        basedir = Path("")
        if "basedir" in params:
            basedir = Path(config["common"].get(params["basedir"], params["basedir"]))
            assert basedir.exists(), f"Base directory {basedir} does not exist."
        info_file = basedir / params["file"]
        repeat = float(params["repeat"])
        filters = params.get("filter", None)
        assert filters is None or isinstance(
            filters, list
        ), f"{filters=} must be a list of strings."
        token_and_path.extend(process_info(info_file, repeat, filters))

    train_data_path = " ".join(token_and_path)
    print(f'export TRAIN_DATA_PATH="{train_data_path}"')

    # Check and print dataset summary
    check_load_dataset(train_data_path, force=force)


if __name__ == "__main__":
    args = argparser.parse_args()
    main(args.yaml_config, force=args.force)
