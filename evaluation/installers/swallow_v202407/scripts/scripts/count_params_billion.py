#!/usr/bin/env python3
"""
count_params_billion.py

Compute the total parameter count of a Hugging Face Transformers model
*without loading the weights*, then print the result rounded **up** to the nearest
billion (10⁹) as “<N> B”.

Usage
-----
$ python count_params_billion.py <model_name_or_path>
# 例:
$ python count_params_billion.py facebook/opt-6.7b
7
"""
import argparse
import math
import sys

import torch
from transformers import AutoConfig, AutoModel


from contextlib import contextmanager
@contextmanager
def meta_tensors():
    orig = torch.nn.Module.register_parameter

    def to_meta(module, name, param):
        if param is not None:
            meta = torch.empty_like(param, device="meta")
            param = torch.nn.Parameter(meta, requires_grad=param.requires_grad)
        orig(module, name, param)

    torch.nn.Module.register_parameter = to_meta
    try:
        yield
    finally:
        torch.nn.Module.register_parameter = orig

def count_parameters(model_name: str) -> int:
    cfg = AutoConfig.from_pretrained(model_name)
    with meta_tensors():                
        model = AutoModel.from_config(cfg)
    return sum(p.numel() for p in model.parameters())

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Count a HF model’s parameters and print ceiling‑billion figure."
    )
    parser.add_argument("model", help="Model name on the Hub or local path")
    args = parser.parse_args(argv)

    total = count_parameters(args.model)
    billions = math.ceil(total / 1e9)
    print(billions)


if __name__ == "__main__":
    main()
