# GENIAC Official Evaluation

This project provides instructions to submit models and tokenizers as W&B artifacts for the GENIAC official evaluation.

## Prerequisites

- Python 3.x installed
- An active W&B account
- Necessary permissions for the W&B entity and project

## Setup

Create a Python virtual environment and install dependencies.

```bash
python3 -m venv venv
. ./venv/bin/activate
pip3 install -r requirements.txt
```

After the installation, login to your W&B account.

```bash
wandb login
```

## Submission

Run the following script to submit a model directory that contains a Hugging Face model checkpoint and corresponding tokenizer.

```bash
python submit.py \
   --entity <wandb entity> \
   --project <wandb project> \
   --model_path <model path> \
   --model_name <model name> \
   --model_version <model version>
```

For example:

```bash
python submit.py \
   --entity nii-geniac \
   --project Llama-2-175B-hf-checkpoints \
   --model_path /home/ext_kiyomaru_nii_ac_jp/checkpoints/megatron-to-hf/Llama-2-172b-hf/iter_0242000 \
   --model_name Llama-2-175B \
   --model_version 0242000
```

