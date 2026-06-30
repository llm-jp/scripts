# v4 MoE B200 SWIFT converter

Convert Megatron `torch_dist` checkpoints to Hugging Face format on the
Sakura/B200 Slurm cluster with `ms-swift`. Self-contained (does not depend on the
ABCI converter dir): the per-model HF configs live here.

Uses `ms-swift==4.3.1`, which the B200 training venv already installs (pinned in
the installer's `src/requirements.txt`), alongside `transformers`, `peft`, and the
CUDA `torch==2.12.1+cu130`. No separate swift install step is needed.

## 1. HF reference (config + tokenizer only — no skeleton weights)

`swift export` builds the HF model structure from `config.json` and fills the
weights from the mcore checkpoint, so the reference directory only needs a
`config.json` and the tokenizer files. **No random HF skeleton weights are
needed** (verified on 291B: config + tokenizer is enough).

Build the reference dir:

```bash
REF=/path/to/hf_ref            # e.g. .../0279_v4-291b-a27b-hf-ref
mkdir -p "$REF"
cp v4-291b-a27b/config.json "$REF"/config.json          # per-model config (this dir)
# tokenizer files (reuse an existing v4 HF ref, same tokenizer for every v4 model):
cp /home/taishi/experiments/0279_v4-moe-32b-a3.8b-hf-ref-config/{special_tokens_map.json,tokenizer_config.json,tokenizer.model} "$REF"/
```

Per-model configs are derived from the Qwen3-MoE shape; `v4-291b-a27b/config.json`
is the 332B shape reduced to `num_hidden_layers=70` with `num_key_value_heads=8`
(GQA=8).

## 2. Submit the conversion

```bash
sbatch --nodelist=<node> \
  --export=ALL,\
VENV_DIR=/home/taishi/envs/megatron-lm-b200,\
MEGATRON_CKPT_STEP=100,\
MODELSCOPE_CACHE=/path/to/modelscope_cache,\
MCORE_PATH=/path/to/model_dir/checkpoints,\
HF_REF_WEIGHT=$REF,\
TEMPLATE=qwen3,\
HF_OUTPUT_DIR=/path/to/hf_output \
  sbatch_convert.sh
```

Output lands in `HF_OUTPUT_DIR/iter_XXXXXXX` (safetensors shards + `config.json`,
`model.safetensors.index.json`, tokenizer, `args.json`). The script builds a
temporary mcore root pointing `latest_checkpointed_iteration.txt` at the step, so
the source checkpoint dir is untouched. One GPU is enough for the conversion
itself (it is a weight remap, not inference).
