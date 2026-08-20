import argparse
import csv
import json
from pathlib import Path


DEFAULT_PROMPT_PREFIX = "以下の質問に対して50文字以内で回答してください。"


def convert_row(row, prompt_number):
    question = row["Question"]
    prompt = DEFAULT_PROMPT_PREFIX + question

    item = dict(row)
    item.update({
        "prompt_number": prompt_number,
        "input": prompt,
        "prompt": prompt,

        # Lowercase aliases make your existing evaluate.py keep these fields.
        "type": row.get("Type", ""),
        "category": row.get("Category", ""),
        "question": question,
    })
    return item


def main():
    parser = argparse.ArgumentParser(
        description="Convert JTruthfulQA.csv to benchmark_data/jtruthfulqa.json."
    )
    parser.add_argument("--input", default="benchmark_data/JTruthfulQA.csv")
    parser.add_argument("--output", default="benchmark_data/jtruthfulqa.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if args.limit is not None:
        rows = rows[: args.limit]

    items = [
        convert_row(row, prompt_number=i)
        for i, row in enumerate(rows, start=1)
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(items)} items to {output_path}")


if __name__ == "__main__":
    main()
