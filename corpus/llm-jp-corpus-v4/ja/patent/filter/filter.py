import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from tqdm.auto import tqdm

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


def is_match_patterns(text: str, patterns: list[str]):
    for _pattern in patterns:
        if _pattern in text:
            return True
    return False


def process_file(input_file: Path, output_dir: Path):
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
                if not do_write and is_match_patterns(_inner_line, extract_range[0]):
                    # start condition
                    do_write = True
                if do_write:
                    if len(extract_range) == 2 and is_match_patterns(
                        _inner_line, extract_range[1]
                    ):
                        # end condition
                        do_write = False
                        continue
                    write_text.append(_inner_line)
            assert (
                len(write_text) > 0
            ), "Document have no extract feature.\n\tFile: {}\n\tLine: {} \n\tFeature:{}".format(
                input_file, i, extract_range[0]
            )
            write_dict["text"] = "\n".join(write_text)
            f_write.write(json.dumps(write_dict, ensure_ascii=False)+"\n")


def main(input_dir: Path, output_dir: Path, worker=16):
    logger.debug(f"{input_dir=}")
    logger.debug(f"{output_dir=}")
    all_files = [
        _f for _f in input_dir.glob("*.jsonl") if _f.stem[5:] in EXTRACT_FEATURES
    ]
    process_file_with_outdir = partial(process_file, output_dir=output_dir)
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

    main(input_dir, output_dir, args.worker)
