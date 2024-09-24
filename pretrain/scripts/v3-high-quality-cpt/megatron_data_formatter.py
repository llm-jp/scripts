"""
This module processes dataset information from CSV files, adjusts token counts, 
and filters entries based on a YAML configuration file. It loads dataset paths, 
calculates total token sizes, and checks for duplicates, with optional detailed output.

Usage:
    python3 megatron_data_formatter.py {yaml_config_file} [-f] [-d]

Arguments:
    yaml_config_file    Path to the YAML configuration file.
    -f, -force          Ignore duplicate dataset entries.
    -d, -display        Display detailed dataset statistics including token sizes and paths.
"""

import csv
from argparse import ArgumentParser
from pathlib import Path

import yaml

# Argument parser setup
argparser = ArgumentParser()
argparser.add_argument("yaml_config")
argparser.add_argument(
    "-f",
    "--force",
    action="store_true",
    help="Ignore duplicate dataset entries",
)
argparser.add_argument(
    "-d",
    "--display",
    action="store_true",
    help="Display detailed dataset statistics including token sizes and paths",
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
        include_chars (list[str], optional): List of filter patterns. Defaults to an empty list.

    Returns:
        list[str]: List of formatted dataset strings (token size and data path).
    """
    data_paths = []
    with open(info_file, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            token_count = int(int(row[2]) * repeat + 0.5)
            data_path = row[1]
            # Filter by filters if provided, otherwise add the path
            if filters is None or any(pattern in data_path for pattern in filters):
                data_paths.append(f"{token_count} {data_path}")
    return data_paths


def check_load_dataset(
    train_data_path: str, force: bool = False, display_details: bool = False
) -> None:
    """
    Checks and displays the dataset details including token sizes and file names.
    Optionally detects and handles duplicate entries.

    Args:
        train_data_path (str): The TRAIN_DATA_PATH string containing dataset paths and token sizes.
        force (bool, optional): If True, ignore duplicates and continue. Defaults to False.
        display_details (bool, optional): If True, display detailed token size breakdown. Defaults to False.
    """
    total_token_size = 0
    file_checker = set()

    if display_details:
        lang_total_tokens: dict[str, int] = {}
        print(f"{'Lang':<5} {'File Name':<20} {'Token Size':>15}")

    data_entries = train_data_path.strip().split()
    for i in range(0, len(data_entries), 2):
        token_size = int(data_entries[i])
        data_path = Path(data_entries[i + 1])
        lang = data_path.parent.name
        file_name = data_path.stem
        combination = f"{lang}/{file_name}"

        if combination in file_checker:
            common_msg = f"Duplicate entry for {combination}."
            if not force:
                raise ValueError(f"{common_msg} Exiting.")
            else:
                print(f"Warning: {common_msg}")
        else:
            file_checker.add(combination)

        if display_details:
            print(f"{lang:<5} {file_name:<20} {token_size:>15,}")
            lang_total_tokens[lang] = lang_total_tokens.get(lang, 0) + token_size

        total_token_size += token_size

    if display_details:
        print("\nSummary:")
        for lang, total in lang_total_tokens.items():
            print(f"{lang:<5} {total:>15,}")
        print(f"ALL   {total_token_size:>15,}")

    print(f"export TOTAL_TOKEN_SIZE={total_token_size}")


def main(config_path: str, force: bool = False, display_details: bool = False) -> None:
    """
    Main function to process datasets as defined in the configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.
        force (bool, optional): If True, ignore duplicates. Defaults to False.
        display_details (bool, optional): If True, display detailed token size breakdown. Defaults to False.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Process each dataset from the config
    token_and_path: list[str] = []
    for params in config["datasets"].values():
        basedir = Path("")
        if "basedir" in params:
            if params["basedir"] in config["common"]:
                basedir = Path(config["common"][params["basedir"]])
                assert basedir.exists(), f"Base directory {basedir} does not exist."
            elif Path(params["basedir"]).exists():
                basedir = Path(params["basedir"])
            else:
                raise ValueError(f"Invalid basedir: {params['basedir']}")

        info_file = basedir / Path(params["file"])
        repeat = float(params["repeat"])
        filters = params.get("filter", None)
        filters = [filters] if isinstance(filters, str) else filters
        assert filters is None or isinstance(
            filters, list
        ), f"{filters=} must be a string or list of strings."
        token_and_path.extend(process_info(info_file, repeat, filters))

    train_data_path = " ".join(token_and_path)
    print(f'export TRAIN_DATA_PATH="{train_data_path}"')

    # Check and print dataset summary
    check_load_dataset(train_data_path, force=force, display_details=display_details)


if __name__ == "__main__":
    args = argparser.parse_args()
    main(args.yaml_config, force=args.force, display_details=args.display)
