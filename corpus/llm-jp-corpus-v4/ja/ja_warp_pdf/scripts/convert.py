"""Remove intra-sentence line breaks from text."""

import argparse
import concurrent
import concurrent.futures
import json
import logging
import pathlib
from typing import Iterator, TextIO

import bunkai
import tqdm
import torch

logger = logging.getLogger(__name__)

torch.set_num_threads(1)

root = pathlib.Path(__file__).parent.parent
model_path = root / "bunkai_model"
senter = bunkai.Bunkai(path_model=model_path)


def split_text_by_length(text: str, max_length: int = 20) -> list[str]:
    """Split text by length.

    Args:
        text (str): Input text.
        max_length (int, optional): Maximum length of sentence. Defaults to 30.

    Returns:
        list[str]: List of sentences.
    """
    if not text:
        return [""]

    chunks: list[str] = []
    for i in range(0, len(text), max_length):
        chunks.append(text[i : i + max_length])
    return chunks


def split_text_by_bunkai(text: str) -> list[str]:
    """Split text by Bunkai.

    Args:
        text (str): Input text.

    Returns:
        list[str]: List of sentences.
    """
    if not text:
        return [""]
    return [s.replace("▁", "\n") for s in senter(text.replace("\n", "▁"))]


def remove_intra_sentence_line_breaks(text: str) -> str:
    """Remove intra-sentence line breaks.

    Args:
        text (str): Input text.

    Returns:
        str: Processed text.
    """
    num_leading_newlines = len(text) - len(text.lstrip("\n"))
    num_trailing_newlines = len(text) - len(text.rstrip("\n"))
    return (
        "\n" * num_leading_newlines
        + text.replace("\n", "")
        + "\n" * num_trailing_newlines
    )


def process_line(line: str) -> str:
    """Process line.

    Args:
        line (str): Input line.

    Returns:
        str: Processed line.
    """
    dat = json.loads(line)

    text = dat["text"]
    new_text = ""
    # Split into small chunks to avoid long processing time
    for chunk in split_text_by_length(text):
        # Skip sentence splitting by bunkai if there is no line break
        # as it aims to remove intra-sentence line breaks
        if "\n" not in chunk:
            new_text += chunk
            continue

        for chunk in split_text_by_bunkai(chunk):
            new_text += remove_intra_sentence_line_breaks(chunk)

    assert text.replace("\n", "") == new_text.replace("\n", "")

    dat["text"] = new_text

    return json.dumps(dat, ensure_ascii=False) + "\n"


def process_lines(lines: list[str]) -> str:
    """Process lines.

    Args:
        lines (list[str]): Input lines.

    Returns:
        str: Processed lines.
    """
    ret: str = ""
    for line in lines:
        try:
            ret += process_line(line)
        except Exception as e:
            logger.error(f"Error: {e}")
            ret += line
    return ret


def buffered_read(file: TextIO, buffer_size: int = 32) -> Iterator[list[str]]:
    """Buffered read.
    
    Args:
        file: File object.
        buffer_size: Buffer size.
    
    Yields:
        str: Line.
    """
    lines: list[str] = []
    for line in file:
        lines.append(line)
        if len(lines) == buffer_size:
            yield lines
            lines = []
    if lines:
        yield lines


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser("Remove intra-sentence line breaks from text.")
    parser.add_argument("--input-file", type=str, required=True, help="Input file.")
    parser.add_argument("--output-file", type=str, required=True, help="Output file.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers.")
    parser.add_argument("--buffer-size", type=int, default=32, help="Buffer size.")
    args = parser.parse_args()

    with (
        open(args.input_file, "rt", encoding="utf-8") as fin,
        open(args.output_file, "wt", encoding="utf-8") as fout,
    ):
        with concurrent.futures.ProcessPoolExecutor(args.num_workers) as executor:
            futures = []
            for lines in buffered_read(fin, buffer_size=args.buffer_size):
                futures.append(executor.submit(process_lines, lines))
            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
            ):
                fout.write(future.result())


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    main()
