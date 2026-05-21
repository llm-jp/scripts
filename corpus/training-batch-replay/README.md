# Training Batch Replay

Replay Megatron-LM pretraining instances without initializing distributed training or loading a model.
By default, the launcher dumps the whole training loop as gzip-compressed JSONL, one instance per line.

## Prerequisites

Prepare an environment for Megatron-LM pretraining, and set up the training data and tokenizer.
Follow our [scripts](https://github.com/llm-jp/scripts/tree/main/pretrain/installers) repository for detailed instructions.

## Local

Run from the installed environment root:

```bash
cd /model/experiments/0342_llm-jp-4-corpus-dump/environment
bash ../scripts/corpus/training-batch-replay/replay_training_batch.sh
```

`replay_training_batch.sh` activates `venv`, builds the Megatron dataset indices,
and writes replayed records.
The default output is `outputs/global-batches.jsonl.gz`.

For a small smoke test, replay one global batch and write it to a separate file:

```bash
ITER_INDEX=0 \
OUTPUT=outputs/test-iter0-datapath.jsonl.gz \
bash ../scripts/corpus/training-batch-replay/replay_training_batch.sh
```

This writes 1024 JSONL records to `outputs/test-iter0-datapath.jsonl.gz` with the default `GLOBAL_BATCH_SIZE=1024`.

## Common Overrides

```bash
OUTPUT=outputs/train.jsonl.gz \
bash ../scripts/corpus/training-batch-replay/replay_training_batch.sh
```

Useful variables:

- `ENV_DIR`: environment root. Defaults to the current directory.
- `MEGATRON_PATH`: Megatron-LM path. Defaults to `src/Megatron-LM`.
- `ITER_INDEX`: replay a single training iteration/global batch index.
- `ITER_START`: first training iteration to replay, inclusive. Defaults to the beginning of the training loop when `ITER_END` is set.
- `ITER_END`: training iteration end, exclusive. Defaults to the end of the training loop when `ITER_START` is set.
- `TRAIN_STEPS`, `GLOBAL_BATCH_SIZE`, `SEQ_LENGTH`, `SEED`: reference training
  parameters.
- `DATA_PATH`, `TOKENIZER_MODEL`, `DATA_CACHE_PATH`: data and tokenizer
  settings. `DATA_PATH` is passed directly to Megatron-LM's `--data-path`.
- `OUTPUT_QUEUE_SIZE`: records buffered between dataset reading and JSONL gzip
  writing. Defaults to `256`; use `0` to disable pipelined writing.
- `GZIP_COMPRESSLEVEL`: gzip compression level. Defaults to `9`; lower values
  trade larger files for faster compression.
- `PROGRESS`, `PROGRESS_INTERVAL`: progress reporting controls. `PROGRESS`
  defaults to `1`; set it to `0` to disable. `PROGRESS_INTERVAL` defaults to
  `5` seconds.
- `EXTRA_ARGS`: additional arguments passed to `replay_training_batch.py`.

## Extracting From A Training Task

Use `extract_replay_config.py` to source a task's `train_data.sh` and
`params.sh`, then emit a shell config file compatible with both
`replay_training_batch.py` and `replay_training_batch.sh`. The only supported
output format is `sh`, and it is the default.

```bash
python ../scripts/corpus/training-batch-replay/extract_replay_config.py \
  /groups/gcg51557/experiments/0213_v4-8b/tasks/v4-8b \
  --output replay_config.sh
```

Then pass the generated config to the Python replay entrypoint:

```bash
python ../scripts/corpus/training-batch-replay/replay_training_batch.py \
  --config replay_config.sh \
  --iter-index 0 \
  --output outputs/iter_0000000.jsonl.gz
```

Or run the full task-oriented flow:

```bash
ENV_DIR=/groups/gcg51557/experiments/0213_v4-8b/env \
OUTPUT_DIR=/groups/gcg51557/experiments/0342_llm-jp-4-corpus-dump/replayed_batches/v4-8b \
bash ../scripts/corpus/training-batch-replay/replay_training_task.sh \
  /groups/gcg51557/experiments/0213_v4-8b/tasks/v4-8b
```

`replay_training_task.sh` writes the generated config to
`OUTPUT_DIR/replay_config.sh` and uses `OUTPUT_DIR/cache` as the default
`DATA_CACHE_PATH`.

The Python entrypoint also supports the same range directly:

```bash
python ../scripts/corpus/training-batch-replay/replay_training_batch.py \
  --megatron-path src/Megatron-LM \
  --iter-start 100 \
  --iter-end 110 \
  --tokenizer-model src/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model \
  --data-path \
    2563804308 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/wiki_0000.jsonl_text_document \
    1826105478 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/kaken_0000.jsonl_text_document \
  --output outputs/train-iter100-110.jsonl.gz
```

The default `DATA_PATH` used by `replay_training_batch.sh` is:

```bash
2563804308 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/wiki_0000.jsonl_text_document 1826105478 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/kaken_0000.jsonl_text_document
```

## Output

Each output line includes `iteration`, `dataset_id`, `dataset_path`, and
`dataset_sample_index`, so records can be traced back to the selected source
dataset in the blend. It also includes `indexed_dataset_spans`, which records the
zero-based `document_id`, token `offset`, and token `length` consumed from each
original Megatron `.idx`/`.bin` instance. If a replayed training sample spans
multiple instances, the spans list contains all of them in sample order.

The first record from the smoke test above has this shape:

```json
{
  "iteration": 0,
  "token_ids": [60220, 29421, 29871, 29421, 30182, 64842, 32171, 30701],
  "text": "シュ・オ・モワーヌやサヴニエール＝クーレ・ド・セランをはじめ、高価なものもある。\n アンジュ (Anjou) - 赤・白・ロゼが作られているが、ロゼが有名。ほか...",
  "dataset_id": 0,
  "dataset_path": "/data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/wiki_0000.jsonl_text_document",
  "dataset_sample_index": 0,
  "indexed_dataset_spans": [
    {"document_id": 701518, "offset": 317, "length": 594},
    {"document_id": 1259311, "offset": 0, "length": 1455}
  ]
}
```

The replay records do not include the top-level blended dataset index or the sample position within a global batch.
