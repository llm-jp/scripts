"""Remove intra-sentence line breaks from text."""

import argparse
import concurrent
import concurrent.futures
import json
import logging
import os
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


def split_text_by_newline(text: str, window: int = 5) -> list[str]:
    """Split text into chunks so that:
        - Concatinating all chunks gives the original text.
        - Each newline character has `window` characters before and after it at least, except for the first and last chunks.

    Args:
        text (str): Input text.

    Returns:
        list[str]: List of chunks.

    Example:
        >>> list(split_text_by_newline("Hello World\n"))
        ["Hello ", "World\n"]
        >>> list(split_text_by_newline("Hello\nWorld"))
        ["Hello\nWorld"]
        >>> list(split_text_by_newline("Hello\nWorld\n"))
        ["Hello\nWorld\n"]
        >>> list(split_text_by_newline("Hello\nWorld\nHello\nWorld\n"))
        ["Hello\nWorld\nHello\nWorld\n"]
        >>> list(split_text_by_newline("CONTEXT|Hello\nWorld\nHello\nWorld|CONTEXT"))
        ["CONTEXT|", "Hello\nWorld\nHello\nWorld", "|CONTEXT"]
    """
    if "\n" not in text:
        return [text]

    chunks: list[str] = []
    chunk: str = ""
    newline_pos: int = -1
    for i, char in enumerate(text):
        if char == "\n":
            newline_pos = i
            if len(chunk) > window and "\n" not in chunk:
                chunks.append(chunk[:-window])
                chunk = chunk[-window:]
        if newline_pos != -1 and i - newline_pos == window + 1:
            chunks.append(chunk)
            chunk = ""
            newline_pos = -1
        chunk += char
    if chunk:
        chunks.append(chunk)
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
    if all(char == "\n" for char in text):
        return text
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

    text: str = dat["text"]
    new_text: str = ""
    for chunk in split_text_by_newline(text, window=5):
        # Skip sentence splitting by bunkai if there is no line break
        # as it aims to remove intra-sentence line breaks
        if "\n" not in chunk:
            new_text += chunk
            continue

        # Skip long chunks as they are usually so noisy that
        # bunkai will not work well
        if len(chunk) > 20:
            new_text += chunk
            continue

        for sent in split_text_by_bunkai(chunk):
            new_text += remove_intra_sentence_line_breaks(sent)

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


def buffered_read(
    file: TextIO,
    processed_ids: set[str],
    buffer_size: int = 32,
) -> Iterator[list[str]]:
    """Buffered read.

    Args:
        file: File object.
        processed_ids: Processed IDs.
        buffer_size: Buffer size.

    Yields:
        str: Line.
    """
    lines: list[str] = []
    for line in file:
        dat = json.loads(line)
        if dat["docId"] in processed_ids:
            continue
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
    parser.add_argument("--num-workers", type=int, default=-1, help="Number of workers.")
    parser.add_argument("--buffer-size", type=int, default=32, help="Buffer size.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file.")
    args = parser.parse_args()

    num_workers = args.num_workers if args.num_workers != -1 else os.cpu_count()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Create an empty file if overwrite is True
    if args.overwrite:
        with open(args.output_file, "wt", encoding="utf-8"):
            pass

    # Get processed lines if overwrite is False
    processed_ids: set[str] = set()
    if not args.overwrite and os.path.exists(args.output_file):
        with open(args.output_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                dat = json.loads(line)
                processed_ids.add(dat["docId"])

    with (
        open(args.input_file, "rt", encoding="utf-8") as fin,
        open(args.output_file, "wt", encoding="utf-8") as fout,
    ):
        with concurrent.futures.ProcessPoolExecutor(num_workers) as executor:
            futures = []
            for lines in buffered_read(
                fin,
                processed_ids=processed_ids,
                buffer_size=args.buffer_size,
            ):
                futures.append(executor.submit(process_lines, lines))
            for future in tqdm.tqdm(futures):
                fout.write(future.result())


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    main()
