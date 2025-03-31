import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

from tqdm import tqdm


class Args:
    worker: int
    input_dir: str
    output_dir: str


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--worker", type=int, default=32)
    return parser


class FilePathCreator:
    def __init__(self) -> str:
        self.counter = 0
        self.prev_mid_path = None

    def get_mid_path(self, path: str) -> str:
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
        mid_path = self.get_mid_path(path)
        if mid_path != self.prev_mid_path:
            self.counter = 0
        self.prev_mid_path = mid_path
        new_file = f"{self.counter:04d}.jsonl.gz"
        self.counter += 1
        return Path(mid_path) / new_file


def normalize_jsonl(data: dict, add_file_path: bool = False):

    # original meta info is stored in data["metadata"]["meta"] if exist
    meta: dict = data.get("metadata", {}).get("meta", {})

    # root keys except "text" and "metadata" and "id"
    # "metadata" and "id" are automatically added on minhash deduplication
    meta_other1 = {k: v for k, v in data.items() if k not in ["text", "metadata", "id"]}

    dedup_meta_keys = ["minhash_cluster_id", "minhash_cluster_size", "token_count"]
    # keys in "metadata", but except {dedup_meta_keys} and "file_path" and ""
    # Omitted keys are automatically added on minhash deduplication
    meta_other2 = {
        k: v
        for k, v in data["metadata"].items()
        if k not in (["file_path", "meta"] + dedup_meta_keys)
    }
    # all keys are assumed to be different
    assert len(set(meta.keys()) & set(meta_other1.keys())) == 0
    assert len(set(meta.keys()) & set(meta_other2.keys())) == 0
    assert len(set(meta_other1.keys()) & set(meta_other2.keys())) == 0

    # store meta info on deduplication in other keys
    if add_file_path:
        dedup_meta_keys.append("file_path")
    dedup_meta = {k: v for k, v in data["metadata"].items() if k in dedup_meta_keys}

    new_meta = meta | meta_other1 | meta_other2 | {"dedup_meta": dedup_meta}

    return {"text": data["text"], "meta": new_meta}


def convert_file(input_file: Path, output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    assert not output_file.exists(), f"{output_file} exists!"
    # Add file_path keys if original data is from ja_fineweb
    # This is because the meta info is comes from original data
    add_file_path = False
    if output_file.parts[3] == "ja_fineweb-2":
        add_file_path = True
    with gzip.open(input_file, "rt") as f_read, gzip.open(output_file, "wt") as f_write:
        for line in f_read:
            data = json.loads(line)
            normalized = normalize_jsonl(data, add_file_path)
            f_write.write(json.dumps(normalized, ensure_ascii=False) + "\n")


class IO_File(NamedTuple):
    input_file: Path
    output_file: Path


def setup_io(input_files: Path, output_dir: Path) -> list[IO_File]:
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


def main(input_dir: str, output_dir: str, worker: int):
    input_files = list(sorted(Path(input_dir).rglob("*.jsonl.gz")))
    io_list = setup_io(input_files, Path(output_dir))

    with ProcessPoolExecutor(max_workers=worker) as executor:
        futures = [executor.submit(convert_file, *_io) for _io in io_list]
        for future in tqdm(as_completed(futures), total=len(io_list), ncols=0):
            try:
                future.result()
            except Exception as e:
                print(f"Worker error: {e}")
                raise


if __name__ == "__main__":
    args = setup_parser().parse_args(namespace=Args)
    assert not Path(args.output_dir).exists(), f"{args.output_dir} is exists!"
    main(args.input_dir, args.output_dir, args.worker)
