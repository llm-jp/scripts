import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Update the result.json file to match the expected format."
)
parser.add_argument("path", type=Path, help="Path to the result.json file.")
args = parser.parse_args()

with open(args.path, encoding="utf-8") as f:
    data = json.load(f)

data["scores"] = data["evaluation"]["scores"]
data["lang_scores"] = data["evaluation"]["lang_scores"]

with open(args.path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
