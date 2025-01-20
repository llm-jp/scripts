# Preprocess fineweb-edu-score-2

This script splits the [HuggingFaceFW/fineweb-edu-score-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2) dataset into `int(score * n).jsonl.gz`.  
Here, `n` (default is 10) can be set to any desired value.

## Install

```bash
pip install datatrove[all]
```

## Usage

### Input directory

```
./fineweb-edu-score-2/data
├── CC-MAIN-2013-20
│   ├── train-00000-of-00058.parquet
│   ├── train-00001-of-00058.parquet
│   ├── ...
│   └── train-00057-of-00058.parquet
├── CC-MAIN-2015-27
├── CC-MAIN-2017-09
├── CC-MAIN-2018-22
├── ...
└── CC-MAIN-2024-51
```

### Output directory

```
./en_fineweb-edu-score-2
├── CC-MAIN-2013-20
│   ├── 15.jsonl.gz
│   ├── 16.jsonl.gz
│   ├── ...
│   └── 42.jsonl.gz
├── CC-MAIN-2015-27
├── CC-MAIN-2017-09
├── CC-MAIN-2018-22
├── ...
└── CC-MAIN-2024-51
```

### Command

```bash
python src/main.py \
    --fineweb-edu-dir Path/to/fineweb-edu-score-2/data \
    --output-dir ./en_fineweb-edu-score-2 \
    --n 10 \
    --cache-size 3000
```
