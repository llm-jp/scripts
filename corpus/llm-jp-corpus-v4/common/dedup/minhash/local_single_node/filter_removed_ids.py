#!/usr/bin/env python
import bz2
import json
import os
from pathlib import Path
import sys
from typing import Set

from .json_format import adapter

removed_id_list: Set[str] = set()


def get_id(record: dict) -> str:
    adapted = adapter(None, record, "", 0)
    return adapted["id"]


def get_removed_id_list_path() -> Path:
    results_dir = Path(os.environ.get(
        "RESULTS_DIR",
        str(Path(__file__).parent / "test_results")))

    return results_dir / "removed_id_list.txt"


def read_removed_id_list():
    global removed_id_list
    with open(get_removed_id_list_path(), "rt") as fin:
        for line in fin:
            removed_id_list.add(line.strip())


def filter_removed_records():
    global removed_id_list

    for line in sys.stdin:
        record = json.loads(line)
        doc_id = get_id(record)
        if doc_id in removed_id_list:
            continue

        print(line, end="")


if __name__ == "__main__":
    read_removed_id_list()
    filter_removed_records()
