# moe-converter-abci3

This repository contains scripts for converting Megatron model checkpoints to Hugging Face format for MoE (Mixture of Experts) models.

## Directory Structure

The expected directory structure is as follows:

```
/
├── convert.sh                 # Main PBS job script for running conversions
├── scripts/
│   └── sakura/
│       └── ckpt/
│           └── mcore_to_hf_mixtral.py  # Conversion script
└── src/
    └── llm-jp-tokenizer/
        └── hf/
            └── ver3.0/
                └── llm-jp-tokenizer-100k.ver3.0b2/  # Tokenizer files
```

## Configuration

Before running the conversion, make sure to check and update the following paths in `convert.sh`:

1. `MEGATRON_PATH="/path/to/default/megatron"`
   - Update this to point to your Megatron-LM installation

2. `TOKENIZER_DIR=src/llm-jp-tokenizer/hf/ver3.0/llm-jp-tokenizer-100k.ver3.0b2/`
   - Verify that this path contains all necessary tokenizer files

3. Ensure that `scripts/sakura/ckpt/mcore_to_hf_mixtral.py` exists
   - This is the main conversion script that will be executed

4. Update the PBS job name to match your experiment number:
   - Change `#PBS -N xxx_conv` to include your specific experiment identifier
   - For example: `#PBS -N 0134_conv`

## Usage

To submit the conversion job to the ABCI queue:

```bash
# Example usage
bash qsub.sh /path/to/source /path/to/target 0 10000 1000 4 2 1 convert.sh
```

The script will convert checkpoints from MCore format to Hugging Face format for specified iterations, copying the tokenizer files to each converted checkpoint directory.

## Required Environment Variables

The following environment variables must be set when submitting the job:

- `START_ITER`: Starting iteration number
- `END_ITER`: Ending iteration number
- `STEP_SIZE`: Step size between iterations
- `BASE_LOAD_PATH`: Base path for loading source checkpoints
- `BASE_SAVE_PATH`: Base path for saving converted checkpoints
- `TARGET_TP_SIZE`: Target tensor parallel size
- `TARGET_PP_SIZE`: Target pipeline parallel size
- `TARGET_EP_SIZE`: Target expert parallel size

## Output

Converted checkpoints will be saved in `${BASE_SAVE_PATH}/iter_XXXXXXX/` directories with tokenizer files copied to each directory.
