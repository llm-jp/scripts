# Script to apply average merging to Hugging Face models.
# This script can work with the pretrain Python environment.
# 
# Usage:
#     python merge.py \
#         --source-models /path/to/model1 /path/to/model2 \
#         --output-model /path/to/output_model

import argparse
import json
import logging
import pathlib
import shutil

import safetensors
import safetensors.torch
import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge Hugging Face models using average merging."
    )
    p.add_argument(
        "--source-models",
        type=pathlib.Path,
        nargs="+",
        required=True,
        help=(
            "Paths to the directory of input models to be merged."
            " All models should be in the same format and have compatible parameters."
        ),
    )
    p.add_argument(
        "--source-weights",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Weights for each source model. If not provided, "
            "all models will be treated equally (weight = 1)."
        ),
    )
    p.add_argument(
        "--output-model",
        type=pathlib.Path,
        required=True,
        help="Path to the output model directory.",
    )

    return p.parse_args()


def iter_params(model_path: pathlib.Path) -> tuple[str, torch.Tensor]:
    """
    Iterate through the parameters of a model stored in a .safetensors file.
    
    Args:
        model_path (pathlib.Path): Path to the .safetensors file.

    Yields:
        tuple[str, torch.Tensor]:
            A parameter name and its corresponding tensor.
    """
    for param_file in model_path.glob("*.safetensors"):
        logging.info(f"  Loading parameters from {param_file}")
        with safetensors.safe_open(param_file, framework="pt", device="cuda:0") as file:
            for key in file.keys():
                yield key, file.get_tensor(key)


def main():
    args = parse_args()

    # Initialize a dictionary to hold the sum of parameters
    param_sums = {}
    model_count = len(args.source_models)
    
    if model_count == 0:
        raise ValueError("No input models provided for merging.")

    logging.info(f"Source models: {args.source_models}")

    # Check weights
    if args.source_weights is None:
        logging.info("No source weights provided, treating all models equally.")
        args.source_weights = [1.0] * model_count
    else
        if len(args.source_weights) != model_count:
            raise ValueError(
                f"Number of source weights ({len(args.source_weights)}) "
                f"does not match number of source models ({model_count})."
            )
        if any(weight <= 0 for weight in args.source_weights):
            raise ValueError("All source weights must be positive.");

    logging.info(f"Source weights: {args.source_weights}")

    # Iterate through each model and accumulate the parameters
    for model_path, weight in zip(args.source_models, args.source_weights):
        if not model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        if not model_path.is_dir():
            raise ValueError(f"Model path {model_path} is not a directory.")
        logging.info(f"Processing model: {model_path}")

        for key, tensor in iter_params(model_path):
            if key not in param_sums:
                param_sums[key] = tensor
            else:
                if param_sums[key].shape != tensor.shape:
                    raise ValueError(f"Shape mismatch for key '{key}': "
                                     f"{param_sums[key].shape} vs {tensor.shape}")
                param_sums[key] += tensor
    
    # Average the parameters
    total_weight = sum(args.source_weights)
    for key in param_sums:
        param_sums[key] /= total_weight

    logging.info("Merging completed. Saving the merged model...")
    args.output_model.mkdir(parents=True, exist_ok=True)
    
    # Copy original files other than .safetensors
    for file in args.source_models[0].iterdir():
        if file.suffix != ".safetensors":
            shutil.copy(file, args.output_model / file.name)
    
    # There should be `model.safetensors.index.json` file
    # containing the mapping of parameter names to their destination file names.
    index_file = args.output_model / "model.safetensors.index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"Index file {index_file} does not exist.")
    with index_file.open("r") as f:
        weight_map = json.load(f)["weight_map"]
    
    # Check if the weight map is consistent with the parameters
    if set(weight_map.keys()) != set(param_sums.keys()):
        raise ValueError("Weight map keys do not match the parameter keys.")
    
    # Make inverse mapping for saving
    output_map = {k: [] for k in set(weight_map.values())}
    for k, v in weight_map.items():
        output_map[v].append(k)
    
    metadata = {"format": "pt"}

    # Save all parameters
    for file_name, keys in output_map.items():
        tensors = {k: param_sums[k] for k in keys}
        output_path = args.output_model / file_name
        logging.info(f"  Saving parameters to {output_path}")
        safetensors.torch.save_file(tensors, output_path, metadata=metadata)
    
    logging.info("Merged model saved successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
