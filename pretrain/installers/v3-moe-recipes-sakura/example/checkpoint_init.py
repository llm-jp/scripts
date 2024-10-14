import os
import random

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model_and_config(model_name, save_directory):

    set_seed(1234)

    # Check if the directory already exists
    if os.path.exists(save_directory):
        print(f"Directory {save_directory} already exists. Skipping model saving.")
        return

    # Load config
    config = AutoConfig.from_pretrained(model_name)
    print(f"Config loaded from {model_name}")

    # Create model from config
    model = AutoModelForCausalLM.from_config(config)
    print("Model created from config")

    # Create save directory
    os.makedirs(save_directory)

    # Save model and config
    model.save_pretrained(save_directory)
    config.save_pretrained(save_directory)

    print(f"Model and config have been saved to {save_directory}")


if __name__ == "__main__":
    model_name = "example/config.json"
    save_directory = "Mixtral-llm-jp-v3-8x1.8B-checkpoint_init/"

    save_model_and_config(model_name, save_directory)
