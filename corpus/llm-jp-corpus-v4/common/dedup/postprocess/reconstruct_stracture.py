# Description:
# This script reconstructs the original directory structure and metadata for deduplicated JSONL.gz files.
# It uses the original file paths (stored in metadata) to group and rename the deduplicated files into subfolders
# using 0-based sequential filenames (e.g., 0000.jsonl.gz, 0001.jsonl.gz, ...).
# The metadata is also normalized to preserve both original and deduplication-related information.

import argparse
import gzip
import json
from pathlib import Path
from typing import NamedTuple

from tqdm import tqdm


class Args:
    """Argument container for command-line parsing."""
    worker: int
    input_dir: str
    output_dir: str


def setup_parser():
    """Set up and return the command-line argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--worker", type=int, default=32)
    return parser


class FilePathCreator:
    """
    Helper class to create output file paths that mirror original corpus structure.

    It groups files using metadata information and creates sequential file names.
    """
    def __init__(self) -> str:
        self.counter = 0
        self.prev_mid_path = None

    def get_mid_path(self, path: str) -> str:
        """
        Convert original input file path into a logical middle path for grouping.
        Some known datasets are mapped to a fixed identifier like "ja_fineweb-2".

        Examples:
            >>> get_mid_path("s3://commoncrawl/crawl-data/CC-MAIN-2023/file1")
            "ja_fineweb-2"

            >>> get_mid_path("/model/experiments/0118_dedup_corpusv4_ja/data/subcorpus/warp_pdf_e0/metadata/sample.jsonl.gz")
            "ja_warp_pdf/e0"

            >>> get_mid_path("/model/experiments/0118_dedup_corpusv4_ja/data/subcorpus/sip_comprehensive_pdf/section/sample.jsonl.gz")
            "ja_sip/comprehensive/pdf"
        """
        if "s3://commoncrawl/crawl-data" in path or "/fsx/guilherme/cc2023-50" in path:
            return "ja_fineweb-2"

        original_file_prefix = (
            "/model/experiments/0118_dedup_corpusv4_ja/data/subcorpus/"
        )
        path_sufix = path.replace(original_file_prefix, "")
        path_parts = Path(path_sufix).parts
        assert len(path_parts) >= 3, f"Input path is invalid format: {path}"

        path_root = "ja_" + path_parts[0]
        if "sip_comprehensive_pdf" in path_root:
            return path_root.replace("-", "/")
        elif "warp_pdf" in path_root:
            return path_root.replace("_e", "/e")
        elif len(path_parts) == 3:
            return path_root
        else:
            # len(path_parts)>3
            return "/".join([path_root] + list(path_parts[2:-1]))

    def get_file_path(self, path: str) -> Path:
        """
        Generate a new file path using the normalized middle path and a counter-based filename.
        The counter resets when the middle path changes.
        """
        mid_path = self.get_mid_path(path)
        if mid_path != self.prev_mid_path:
            self.counter = 0
        self.prev_mid_path = mid_path
        new_file = f"{self.counter:04d}.jsonl.gz"
        self.counter += 1
        return Path(mid_path) / new_file


def normalize_jsonl(data: dict, add_file_path: bool = False):
    """
    Normalize the metadata format of a JSONL entry.

    Combines metadata from various levels and relocates deduplication-related fields under `meta["dedup_meta"]`.
    """
    meta: dict = data.get("metadata", {}).get("meta", {})
    meta_other1 = {k: v for k, v in data.items() if k not in ["text", "metadata", "id"]}

    dedup_meta_keys = ["minhash_cluster_id", "minhash_cluster_size", "token_count"]
    # Extract metadata keys excluding the deduplication-related ones
    meta_other2 = {
        k: v
        for k, v in data["metadata"].items()
        if k not in (["file_path", "meta"] + dedup_meta_keys)
    }

    # Ensure no overlapping keys between different metadata sections
    assert len(set(meta.keys()) & set(meta_other1.keys())) == 0
    assert len(set(meta.keys()) & set(meta_other2.keys())) == 0
    assert len(set(meta_other1.keys()) & set(meta_other2.keys())) == 0

    # Store deduplication metadata if required
    if add_file_path:
        dedup_meta_keys.append("file_path")
    dedup_meta = {k: v for k, v in data["metadata"].items() if k in dedup_meta_keys}

    new_meta = meta | meta_other1 | meta_other2 | {"dedup_meta": dedup_meta}

    return {"text": data["text"], "meta": new_meta}


def convert_file(input_file: Path, output_file: Path):
    """
    Read a gzipped JSONL file, normalize each line's metadata, and write to a new gzipped file.
    If the file is from ja_fineweb-2, include the original file path in dedup metadata.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    assert not output_file.exists(), f"{output_file} exists!"

    # Determine if the original file is from ja_fineweb-2 to include additional metadata
    add_file_path = False
    if output_file.parts[3] == "ja_fineweb-2":
        add_file_path = True

    with gzip.open(input_file, "rt") as f_read, gzip.open(output_file, "wt") as f_write:
        for line in f_read:
            data = json.loads(line)
            normalized = normalize_jsonl(data, add_file_path)
            f_write.write(json.dumps(normalized, ensure_ascii=False) + "\n")


class IO_File(NamedTuple):
    """Simple tuple that pairs an input file with its output file path."""
    input_file: Path
    output_file: Path


def setup_io(input_files: Path, output_dir: Path) -> list[IO_File]:
    """
    Prepare a list of IO_File pairs by inspecting metadata from each input file.
    Determines the correct output location and file name based on metadata.
    """
    io_list = []
    file_path_creator = FilePathCreator()
    for _file in tqdm(input_files, ncols=0):
        with gzip.open(_file, "rt") as f:
            line = f.readline()
            data = json.loads(line)
            original_file_path = data["metadata"]["file_path"]
            output_file = file_path_creator.get_file_path(str(original_file_path))
            output_file = Path(output_dir) / output_file
            io_list.append(IO_File(_file, output_file))
    return io_list
