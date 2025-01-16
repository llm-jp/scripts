"""Remove intra-sentence line breaks from text."""

import argparse
import json
import logging
import pathlib

import bunkai
import tqdm

logger = logging.getLogger(__name__)

root = pathlib.Path(__file__).parent.parent
model_path = root / "bunkai_model"
senter = bunkai.Bunkai(path_model=model_path)


def split_text_by_period(text: str) -> list[str]:
    """Split text by period.

    Args:
        text (str): Input text.

    Returns:
        list[str]: List of sentences.
    """
    if not text:
        return [""]

    chunks: list[str] = []
    chunk: str = ""
    for char in text:
        chunk += char
        if char == "。":
            chunks.append(chunk)
            chunk = ""
    if chunk:
        chunks.append(chunk)
    return chunks


def split_text_by_length(text: str, max_length: int = 100) -> list[str]:
    """Split text by length.

    Args:
        text (str): Input text.
        max_length (int, optional): Maximum length of sentence. Defaults to 100.

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
    return list(senter(text))


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
    for chunk in split_text_by_period(text):
        # Split large chunk to avoid long processing time
        for chunk in split_text_by_length(chunk, max_length=100):
            # Skip sentence splitting by bunkai if there is no line break as it is slow
            if "\n" not in chunk:
                new_text += chunk
                continue

            for chunk in split_text_by_bunkai(chunk):
                new_text += remove_intra_sentence_line_breaks(chunk)

    assert text.replace("\n", "") == new_text.replace("\n", "")

    dat["text"] = new_text

    return json.dumps(dat, ensure_ascii=False) + "\n"


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser("Remove intra-sentence line breaks from text.")
    parser.add_argument("--input-file", type=str, required=True, help="Input file.")
    parser.add_argument("--output-file", type=str, required=True, help="Output file.")
    args = parser.parse_args()

    with (
        open(args.input_file, "rt", encoding="utf-8") as fin,
        open(args.output_file, "wt", encoding="utf-8") as fout,
    ):
        for i, line in enumerate(tqdm.tqdm(fin), start=1):
            try:
                line = process_line(line)
            except Exception as e:
                logger.error(f"Error processing line {i}")
                logger.error(e)
            fout.write(line)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    main()
