# Preprocess FineWeb

This script filters out duplicate documents from [HuggingFaceFW/fineweb-edu-score-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb) based on [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) or [HuggingFaceFW/fineweb-edu-score-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2).

## Installation

```bash
pip install datatrove[all]
```

## Usage

### Input directory

```
./fineweb/data
├── CC-MAIN-2013-20
│   ├── 000_00000.parquet
│   ├── 000_00001.parquet
│   ├── ...
│   └── 004_00004.parquet
├── CC-MAIN-2015-27
├── CC-MAIN-2017-09
├── CC-MAIN-2018-22
├── ...
└── CC-MAIN-2024-51
```

### Output directory

```
./en_fineweb
├── CC-MAIN-2013-20
│   ├── 000_00000.jsonl.gz
│   ├── 000_00001.jsonl.gz
│   ├── ...
│   └── 004_00004.jsonl.gz
├── CC-MAIN-2015-27
├── CC-MAIN-2017-09
├── CC-MAIN-2018-22
├── ...
└── CC-MAIN-2024-51
```

### Command

```bash
python src/main.py \
    --fineweb-edu-dir Path/to/fineweb_edu/data \
    --fineweb-dir Path/to/fineweb/data \
    --output-dir ./en_fineweb \
    --num-workers 10 \
    --cache-size 3000
```
