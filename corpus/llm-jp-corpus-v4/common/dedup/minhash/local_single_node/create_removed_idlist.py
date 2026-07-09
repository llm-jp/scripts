#!/usr/bin/env python
import gzip
import json
import os
from pathlib import Path

from .json_format import adapter


def get_id(record: dict) -> str:
    adapted = adapter(None, record, "", 0)
    return adapted["id"]


def iter_remove_records(removed_dir: Path):
    for file in removed_dir.glob("*.jsonl.gz"):
        with gzip.open(file, "rt") as fin:
            for line in fin:
                record = json.loads(line)
                doc_id = get_id(record)
                if doc_id == "":
                    raise RuntimeError(f"No id found in '{line}'")

                yield doc_id


def main():
    results_dir = Path(os.environ.get(
        "RESULTS_DIR",
        str(Path(__file__).parent / "test_results")))
    removed_dir = results_dir / "removed/"

    with open(results_dir / "removed_id_list.txt", "wt") as fout:
        for doc_id in iter_remove_records(Path(removed_dir)):
            print(doc_id, file=fout)


if __name__ == "__main__":
    main()
