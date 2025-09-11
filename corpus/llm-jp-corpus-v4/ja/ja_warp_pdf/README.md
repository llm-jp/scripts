# ja-warp-pdf

Preprocess text extracted from PDFs provided by WARP.

## Environment

- Python 3.12.5

## Installation

Use [rye](https://rye.astral.sh/) to install the dependencies.

```bash
RUSTFLAGS="-A invalid_reference_casting" rye sync
```

Then download the Bunkai sentence splitter model.

```bash
rye run bunkai --model bunkai_model --setup
```

## Usage

### Conversion

This process converts text to remove unnecessary characters.

```bash
rye run python scripts/convert.py --input-file <input-file> --output-file <output-file>
```
