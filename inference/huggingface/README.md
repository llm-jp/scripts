# Inference with HuggingFace Models

This directory provides a minimum script to perform inference with causal language models saved in the HuggingFace format.

## Installation

Run the following command:

```bash
python3 -m venv venv
. ./venv/bin/activate
pip3 install -r requirements.txt
```

## Inference

Run the following script:

```bash
python3 inference.py --model-name-or-path <model name or path>
```

See the command-line help for more options:

```bash
python3 inference.py --help
```
