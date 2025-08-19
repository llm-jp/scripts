## Convert Megatron Torch-Distributed Checkpoints to Hugging Face (CPU)

### Overview
`convert_to_hf_cpu.sh` submits a PBS job to convert a single Megatron-LM torch_distributed checkpoint (MCore) into Hugging Face format on CPU.

### Prerequisites
- A working environment directory (`ENV_DIR`) that contains:
  - Python virtualenv at `ENV_DIR/venv` with CPU-only PyTorch installed (requirements.txt).
  - Megatron-LM checked out at `ENV_DIR/src/Megatron-LM` (the script calls `tools/checkpoint/convert.py`).
- An experiment directory (`EXP_DIR`) that provides the HF tokenizer at:
  - `EXP_DIR/src/llm-jp-tokenizer/hf/ver3.0/llm-jp-tokenizer-100k.ver3.0b2` (modify the qsub_convert_cpu.sh file to use other tokenizers)
- Input checkpoints located at:
  - `TASK_DIR/checkpoints/iter_XXXXXXX` (7-digit zero-padded).

### Usage
```bash
bash scripts/ckpt_converter/convert_to_hf_cpu.sh \
  /path/to/TASK_DIR \
  100000 \
  /path/to/EXP_DIR \
  /path/to/ENV_DIR
```

- `TASK_DIR`: Root of your training run; must contain `checkpoints/iter_XXXXXXX`.
- `ITER`: Integer iteration number (e.g., `100000`). The job will internally format it as `iter_0000000`.
- `EXP_DIR`: Experiment dir that holds `src/llm-jp-tokenizer/hf/ver3.0/llm-jp-tokenizer-100k.ver3.0b2`.
- `ENV_DIR`: Environment dir with `venv` (CPU torch) and `src/Megatron-LM`.

The script queues a single PBS job, passing these values to `qsub_convert_cpu.sh`.