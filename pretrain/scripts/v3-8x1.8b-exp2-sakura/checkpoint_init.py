import argparse
import logging
import random
import re

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def initialize_ffn_weights(size, init_method, current_mean=0, current_std=0.02):
    logger.info("Initializing FFN weights:")
    logger.info(f"  Size: {size}")
    logger.info(f"  Initialization method: {init_method}")
    logger.info(f"  Current mean: {current_mean}")
    logger.info(f"  Current std: {current_std}")

    if init_method == "zero_mean_002std":
        logger.info("  Using zero mean and 0.02 standard deviation")
        return torch.normal(mean=0, std=0.02, size=size)
    elif init_method == "zero_mean_current_std":
        logger.info(f"  Using zero mean and current standard deviation ({current_std})")
        return torch.normal(mean=0, std=current_std, size=size)
    elif init_method == "current_mean_002std":
        logger.info(
            f"  Using current mean ({current_mean}) and 0.02 standard deviation"
        )
        return torch.normal(mean=current_mean, std=0.02, size=size)
    elif init_method == "current_mean_current_std":
        logger.info(
            f"  Using current mean ({current_mean}) and current standard deviation ({current_std})"
        )
        return torch.normal(mean=current_mean, std=current_std, size=size)
    else:
        logger.error(f"Unknown initialization method: {init_method}")
        raise ValueError(f"Unknown initialization method: {init_method}")


def partially_initialize(
    tensor,
    init_indices,
    is_down_proj,
    layer_idx,
    expert_idx,
    init_method,
    share_init_indices,
    ffn_init_ratio,
):
    if is_down_proj:
        init_part = tensor[:, init_indices]
    else:
        init_part = tensor[init_indices, :]

    current_mean = init_part.mean().item()
    current_std = init_part.std().item()
    if is_down_proj:
        init_tensor = initialize_ffn_weights(
            (tensor.size(0), len(init_indices)),
            init_method,
            current_mean=current_mean,
            current_std=current_std,
        ).to(dtype=torch.bfloat16)
        tensor[:, init_indices] = init_tensor
    else:
        init_tensor = initialize_ffn_weights(
            (len(init_indices), tensor.size(1)),
            init_method,
            current_mean=current_mean,
            current_std=current_std,
        ).to(dtype=torch.bfloat16)
        tensor[init_indices, :] = init_tensor

    logger.info(
        f"Layer {layer_idx}, Expert {expert_idx}, {'Down_proj' if is_down_proj else 'Gate_proj/Up_proj'}: "
        f"Original size: {tensor.size()}, "
        f"Initialization method: {init_method}, Share init indices: {share_init_indices}, "
        f"Init ratio: {ffn_init_ratio}, Init size: {len(init_indices)}, "
        f"Init part mean: {current_mean:.4f}, Init part std: {current_std:.4f}"
    )
    logger.info(f"Init indices: {init_indices[:10]}... (showing first 10 elements)")

    return tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_gate_weights(size):
    return torch.normal(mean=0, std=0.02, size=size)


def replace_model_parameters(
    source_model_path,
    target_config_path,
    output_path,
    num_experts,
    num_layers,
    seed,
    init_method,
    share_init_indices,
    ffn_init_ratio,
):
    set_seed(seed)
    logger.info("Starting model parameter replacement process")
    logger.info("Configuration:")
    logger.info(f"  Source model: {source_model_path}")
    logger.info(f"  Target config: {target_config_path}")
    logger.info(f"  Output path: {output_path}")
    logger.info(f"  Number of experts: {num_experts}")
    logger.info(f"  Number of layers: {num_layers}")
    logger.info(f"  Seed: {seed}")
    logger.info(f"  FFN initialization method: {init_method}")
    logger.info(f"  Share initialization indices: {share_init_indices}")
    logger.info(f"  FFN initialization ratio: {ffn_init_ratio}")

    logger.info("Loading source model")
    source_model = AutoModelForCausalLM.from_pretrained(
        source_model_path, torch_dtype=torch.bfloat16
    )
    logger.info("Loading target config")
    target_config = AutoConfig.from_pretrained(target_config_path)
    logger.info("Creating target model from config")
    target_model = AutoModelForCausalLM.from_config(
        target_config, torch_dtype=torch.bfloat16
    )
    source_intermediate_size = source_model.config.intermediate_size
    target_intermediate_size = target_config.intermediate_size
    logger.info(f"Source intermediate size: {source_intermediate_size}")
    logger.info(f"Target intermediate size: {target_intermediate_size}")

    exclude_pattern = r"model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight"
    exclude_layers = set()
    for name in target_model.state_dict().keys():
        if re.match(exclude_pattern, name):
            exclude_layers.add(name)

    base_src = "model.layers.{}.block_sparse_moe.experts.{}"
    base_tgt = "model.layers.{}.mlp"
    replace_mapping = {
        f"{base_src}.w1.weight": f"{base_tgt}.gate_proj.weight",
        f"{base_src}.w2.weight": f"{base_tgt}.down_proj.weight",
        f"{base_src}.w3.weight": f"{base_tgt}.up_proj.weight",
    }

    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()

    for name, param in tqdm(target_state_dict.items(), desc="Replacing parameters"):
        if name not in exclude_layers and name in source_state_dict:
            target_state_dict[name] = source_state_dict[name]
            logger.info(f"Parameter {name} replaced")

    for layer_idx in tqdm(range(num_layers), desc="Initializing gate weights"):
        gate_weight_name = f"model.layers.{layer_idx}.block_sparse_moe.gate.weight"
        if gate_weight_name in target_state_dict:
            target_state_dict[gate_weight_name] = initialize_gate_weights(
                target_state_dict[gate_weight_name].size()
            )
            logger.info(
                f"Gate weight {gate_weight_name} initialized with normal distribution (std=0.02)"
            )
    init_size = int(target_intermediate_size * ffn_init_ratio)
    for layer_idx in tqdm(range(num_layers), desc="Replacing FFN layers"):

        if share_init_indices:
            shared_init_indices = torch.randperm(target_intermediate_size)[:init_size]
            logger.info(
                f"Layer {layer_idx}, Generated shared init indices: {shared_init_indices[:10]}... (showing first 10 elements)"
            )
        for expert_idx in range(num_experts):
            if not share_init_indices:
                init_indices = torch.randperm(target_intermediate_size)[:init_size]
                logger.info(
                    f"Layer {layer_idx}, Expert {expert_idx}, Generated init indices: {init_indices[:10]}... (showing first 10 elements)"
                )
            else:
                init_indices = shared_init_indices

            for target_pattern, source_pattern in replace_mapping.items():
                target_name = target_pattern.format(layer_idx, expert_idx)
                source_name = source_pattern.format(layer_idx)
                if (
                    target_name in target_state_dict
                    and source_name in source_state_dict
                ):
                    source_tensor = source_state_dict[source_name]

                    # Determine if it's down_proj (w2) or not
                    is_down_proj = "down_proj" in source_name
                    logger.info(
                        f"Layer {layer_idx}, Expert {expert_idx}, Original tensor shape: {source_tensor.shape}"
                    )
                    if (
                        source_tensor.size(1 if is_down_proj else 0)
                        > target_intermediate_size
                    ):
                        # Resize the tensor if necessary
                        if is_down_proj:
                            source_tensor = source_tensor[:, :target_intermediate_size]
                        else:
                            source_tensor = source_tensor[:target_intermediate_size, :]

                    initialized_tensor = partially_initialize(
                        source_tensor,
                        init_indices,
                        is_down_proj,
                        layer_idx,
                        expert_idx,
                        init_method,
                        share_init_indices,
                        ffn_init_ratio,
                    )
                    logger.info(
                        f"Layer {layer_idx}, Expert {expert_idx}, Initialized tensor shape: {initialized_tensor.shape}"
                    )
                    target_state_dict[target_name] = initialized_tensor

                    logger.info(f"FFN layer {target_name} replaced with {source_name}")

    target_model.load_state_dict(target_state_dict)
    target_model.save_pretrained(output_path, torch_dtype=torch.bfloat16)
    logger.info(f"Modified model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Replace model parameters")
    parser.add_argument(
        "--ffn_init_method",
        type=str,
        choices=[
            "zero_mean_002std",
            "zero_mean_current_std",
            "current_mean_002std",
            "current_mean_current_std",
        ],
        required=True,
        help="Method for initializing FFN weights",
    )
    parser.add_argument(
        "--share_init_indices",
        action="store_true",
        help="Share initialization indices across experts within each layer",
    )
    parser.add_argument(
        "--ffn_init_ratio",
        type=float,
        default=0.5,
        help="Ratio of initialized weights (0.0 to 1.0)",
    )
    parser.add_argument(
        "--source_model_path", type=str, required=True, help="Path to the source model"
    )
    parser.add_argument(
        "--target_config_path",
        type=str,
        required=True,
        help="Path to the target model config",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the modified model"
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        required=True,
        help="Number of experts in the MoE model",
    )
    parser.add_argument(
        "--num_layers", type=int, required=True, help="Number of layers in the model"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    replace_model_parameters(
        args.source_model_path,
        args.target_config_path,
        args.output_path,
        args.num_experts,
        args.num_layers,
        args.seed,
        args.ffn_init_method,
        args.share_init_indices,
        args.ffn_init_ratio,
    )


if __name__ == "__main__":
    main()
