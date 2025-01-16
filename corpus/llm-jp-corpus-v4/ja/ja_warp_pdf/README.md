# ja-warp-pdf

Preprocess text extracted from PDFs provided by WARP.

## Environment

- Python 3.12.5

## Installation

Use [rye](https://rye.astral.sh/) to install the dependencies.

```bash
rye sync
```

and then download the Bunkai sentence splitter model.

```bash
rye run bunkai --model bunkai_model --setup
```
