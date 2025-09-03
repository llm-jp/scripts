## Convert Megatron Torch-Distributed Checkpoints to Hugging Face (CPU)

### Overview
`convert_to_hf_cpu.sh` submits a PBS job to convert a single Megatron-LM torch_distributed checkpoint (MCore) into Hugging Face format on CPU.

### Prerequisites
- A Python virtualenv (`VENV_DIR`) with CPU-only PyTorch installed (see `requirements.txt`).
- Megatron-LM checked out at `MEGATRON_PATH` (the script calls `tools/checkpoint/convert.py`).
- A Hugging Face tokenizer directory at `HF_TOKENIZER_PATH`.
- Input checkpoints located at:
  - `TASK_DIR/checkpoints/iter_XXXXXXX` (7-digit zero-padded).

How to install anappropriate venv (using uv):

```shell
mkdir cpuenv
cd cpuenv
uv python pin 3.13
uv init
uv venv
uv pip install "torch==2.8.0+cpu" -i https://download.pytorch.org/whl/cpu
uv pip install -r /path/to/ckpt_converter/requirements.txt
```

### Usage
```bash
bash scripts/ckpt_converter/convert_to_hf_cpu.sh \
  /path/to/TASK_DIR \
  100000 \
  /path/to/VENV_DIR \
  /path/to/MEGATRON_PATH \
  /path/to/HF_TOKENIZER_PATH \
  /path/to/OUTPUT_DIR
```

- `TASK_DIR`: Root of training run; must contain `checkpoints/iter_XXXXXXX`.
- `ITER`: Integer iteration number (e.g., `100000`). The job will internally format it as `iter_0000000`.
- `VENV_DIR`: Python virtualenv directory (must contain `bin/activate`).
- `MEGATRON_PATH`: Megatron-LM root directory (must contain `tools/checkpoint/convert.py`).
- `HF_TOKENIZER_PATH`: Path to the Hugging Face tokenizer directory.
- `OUTPUT_DIR`: Destination directory for converted HF checkpoints.

The script queues a single PBS job, passing these values to `qsub_convert_cpu.sh`.
