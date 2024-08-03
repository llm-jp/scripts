# GENIAC Official Evaluation

This document provides instructions to submit models and tokenizers as W&B artifacts for the GENIAC official evaluation.

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
   --model-path <model path> \
   --model-name <model name> \
   --model-version <model version>
```

For example:

```bash
python submit.py \
   --entity nii-geniac \
   --project Llama-2-175B-hf-checkpoints \
   --model-path /home/ext_kiyomaru_nii_ac_jp/checkpoints/megatron-to-hf/Llama-2-172b-hf/iter_0242000 \
   --model-name Llama-2-175B \
   --model-version 0242000
```
