import argparse
import hashlib
import itertools
import json
import logging
import tempfile
from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)


def length_check(line: str) -> int:
    """Perform a length check on the line."""
    return len(line)


def sha512_check(line: str) -> str:
    """Perform a hash check using SHA512 on the line."""
    return hashlib.sha512(line.encode("utf-8")).hexdigest()


def calculate_checks(file_path: Path, checks: list[str]) -> list[dict[str, str | int]]:
    """Perform specified checks on the file and return results."""
    results = list()
    try:
        with file_path.open() as f:
            for line in f:
                data = json.loads(line)
                check_result = dict()
                if "length" in checks:
                    check_result["length"] = length_check(data)
                if "hash" in checks:
                    check_result["sha512"] = sha512_check(data)
                results.append(check_result)
    except Exception as e:
        logging.error(f"Error processing file {file_path}")
        raise e
    return results


def create_temp_workdir(
    input_paths: list[Path],
    output_names: list[str],
    tmp_path: str | None = None,
) -> tuple[tempfile.TemporaryDirectory, dict[Path, Path]]:
    """Create a temporary structure and return the mapping and temporary directory."""

    tmp_base = Path(tmp_path).parent if tmp_path else None
    tmp_prefix = Path(tmp_path).name if tmp_path else None
    temp_dir = tempfile.TemporaryDirectory(prefix=tmp_prefix, dir=str(tmp_base))

    input2tmp = {}
    for _path, _name in zip(input_paths, output_names):
        temp_path = Path(temp_dir) / _name
        temp_path.mkdir(parents=True)
        input2tmp[_path] = _name

    return temp_dir, input2tmp


def write_checks_to_temp(
    file_path: Path, temp_file: Path, checks: list[str]
) -> list[dict[str, str | int]]:
    """Write checks to a temporary file."""
    results = calculate_checks(file_path, checks)
    with temp_file.open("w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    return results


def detect_duplicates(checked_sigs: list[dict[str, str | int]]) -> tuple[dict, set]:
    # {tuple(duplicated values):count}
    duplication_counter = defaultdict(int)
    for _checked_sig in checked_sigs:
        _values = tuple(_checked_sig.values())
        duplication_counter[tuple(_values)] += 1

    duplication_counter = {k: v for k, v in duplication_counter.items() if v > 1}
    return duplication_counter


class judgeDuplication:
    def __init__(self, duplication_counts: dict[tuple, int], char_check: bool):
        self.duplication_counts = duplication_counts
        self.char_check = char_check
        self.deduped_pattern = set()
        self.pattern2char = defaultdict(set)

    def is_uniq(self, values: tuple, char: str = None):
        if self.duplication_counts.get(values, -1) <= 1:
            # Is not duplicated pattern
            return True
        if values not in self.deduped_pattern:
            # First duplicated pattern
            return True
        elif not self.char_check:
            # Check exact match by character
            return False
        elif self.is_same_char(values, char):
            return False
        return True

    def is_same_char(self, values: tuple, char: str) -> bool:
        assert char is not None
        if char in self.pattern2char[values]:
            return True
        else:
            return False

    def add_deduped(self, values: tuple, char: str | None = None):
        if self.duplication_counts.get(values, -1) > 1:
            self.deduped_pattern.add(values)
            if self.char_check:
                self.pattern2char[values].add(char)


def export_results(
    input2tmp: dict[Path, Path],
    tmp_dir: str,
    output_root: str,
    duplication_counts: dict[tuple, int],
    checks: list[str],
    char_check: bool,
) -> None:
    """Export duplicate, and deduplicated files."""
    duplicate_dir = Path(output_root) / "duplicate"
    dedup_dir = Path(output_root) / "dedup"
    judger = judgeDuplication(duplication_counts, char_check)

    for _input_path, _out_name in input2tmp.items():
        tmp_dir = Path(tmp_dir) / _out_name
        for file_path in _input_path.rglob("*"):
            if not (_input_path / file_path).is_file:
                continue
            # set reference
            input_file = _input_path / file_path
            tmp_file = tmp_dir / file_path
            # set output files
            duplicate_file = duplicate_dir / file_path
            dedup_file = dedup_dir / file_path
            # create output dir
            duplicate_file.parent.mkdir(parents=True, exist_ok=True)
            dedup_file.parent.mkdir(parents=True, exist_ok=True)

        with open(input_file) as f_input, open(tmp_file) as f_tmp, \
            open(duplicate_file, "w") as f_dup, open(dedup_file, "w") as f_dedup:  # fmt: skip
            for line_input, line_tmp in zip(f_input, f_tmp):
                input_data: dict = json.load(line_input)
                check_data: dict = json.load(line_tmp)
                _values = tuple([check_data[check_key] for check_key in checks])
                write_data = json.dumps(input_data | check_data)

                if not judger.is_uniq(_values, input_data["text"]):
                    f_dup.write(write_data)
                else:
                    judger.add_deduped(_values, input_data["text"])
                    f_dedup.write(write_data)


def main(
    input_paths: list[Path],
    output_path: Path,
    temp_path: str,
    output_names: list[str],
    checks: list[str],
):
    # checks
    assert len(checks) == len(set(checks))
    char_check = True if "char" in checks else False
    checks = [c for c in checks if c != "char"]

    temp_dir, input2tmp = create_temp_workdir(
        input_paths,
        output_names,
        temp_path,
    )

    futures: list[Future] = []
    with ProcessPoolExecutor(max_workers=16) as executor:
        for _input_path, _temp_path in input2tmp.items():
            for file_path in _input_path.rglob("*"):
                if not (_input_path / file_path).is_file:
                    continue
                _temp_file = Path(temp_dir) / _temp_path / file_path
                _temp_file.parent.mkdir(parents=True, exist_ok=True)
                futures.append(
                    executor.submit(
                        write_checks_to_temp,
                        _input_path / file_path,
                        _temp_file,
                        checks,
                    )
                )

    # all_tmp_path = [Path(temp_dir) / _temp_path for _temp_path in input2tmp.values()]
    duplication_counts = detect_duplicates(
        itertools.chain.from_iterable((_future.result() for _future in futures))
    )
    export_results(
        input2tmp, temp_dir, output_path, duplication_counts, checks, char_check
    )
    temp_dir.cleanup()


if __name__ == "__main__":

    class Args(argparse.Namespace):
        input: list[str]
        output: str
        output_names: list[str]
        checks: list[str]
        tmp: str

    parser = argparse.ArgumentParser(
        description="Detect and handle duplicate JSONL files using multiple checks."
    )
    parser.add_argument(
        "--input", nargs="+", required=True, help="Paths to input files or directories."
    )
    parser.add_argument("--output", required=True, help="Path to the output directory.")
    parser.add_argument(
        "--output-names",
        nargs="+",
        help="Custom names for the temporary structure (default: input file or folder names).",
    )
    parser.add_argument(
        "--checks",
        nargs="+",
        default=["sha512", "length", "char"],
        help="Checks to perform on each line (default: length, sha512, char).",
    )
    parser.add_argument(
        "--tmp",
        required=False,
        default=None,
        help="Base path and prefix for the temporary directory (e.g., /tmp/temp_check_).",
    )

    args = parser.parse_args(namespace=Args)

    # Validate unique output names
    if args.output_names:
        if len(args.input) != len(args.output_names):
            raise ValueError(
                "The number of input paths must match the number of output names."
            )
        if len(set(args.output_names)) != len(args.output_names):
            raise ValueError("Output names must be unique.")

    # Validate input paths
    input_paths = [Path(p) for p in args.input]
    for _path in zip(input_paths):
        if not _path.exists():
            raise FileNotFoundError(f"Input path does not exist: {_path}")

    # Validate output names
    if args.output_names is None:
        output_names = [p.name for p in input_paths]
    else:
        output_names = args.output_names

    seen_names = set()
    for _name in output_names:
        if _name in seen_names:
            raise ValueError(
                f"Duplicate output name detected: {_name}. Please provide unique names."
            )
        seen_names.add(_name)

    main(input_paths, Path(args.output), args.tmp, output_names, args.checks)
