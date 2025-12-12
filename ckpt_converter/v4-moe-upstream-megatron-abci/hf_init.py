import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Initialize HF CausalLM from config and save bf16 weights"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path or HF Hub ID of config"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory to save initialized model",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. load config
    config = AutoConfig.from_pretrained(args.config)

    # 2. build model
    model = AutoModelForCausalLM.from_config(config)

    # 3. cast to bf16
    model.to(torch.bfloat16)

    # 4. save (safetensors)
    model.save_pretrained(args.out, safe_serialization=True)

    print("[OK] Initialized bf16 model saved")
    print(f"     config: {args.config}")
    print(f"     out:    {args.out}")


if __name__ == "__main__":
    main()
