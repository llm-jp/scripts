# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
from joblib import Parallel, delayed
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import (
    MegatronGPTModel,
)
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils import logging
from pytorch_lightning import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer


def convert_layer(model, layer: int, torch_dtype) -> dict:
    converted = {}

    param_to_weights = lambda param: param.to(torch_dtype)  # noqa

    hidden_size: int = model.cfg.hidden_size
    head_num: int = model.cfg.num_attention_heads
    ffn_hidden_size: int = model.cfg.ffn_hidden_size
    num_query_groups: int = model.cfg.get("num_query_groups", head_num)

    head_size: int = hidden_size // head_num
    heads_per_group: int = head_num // num_query_groups
    qkv_total_dim: int = head_num + 2 * num_query_groups

    qkv_weights = model.state_dict()[
        f"model.module.decoder.layers.{layer}.self_attention.linear_qkv.weight"
    ]
    qkv_weights = qkv_weights.reshape([qkv_total_dim, head_size, hidden_size])

    q_slice = torch.cat(
        [
            torch.arange(
                (heads_per_group + 2) * i,
                (heads_per_group + 2) * i + heads_per_group,
            )
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))
    # Example of slices
    # 7b: num_query_groups = head_num = 32,
    # q_slice = [0, 3, 6, 9 , ... 90, 93]
    # k_slice = [1, 4, 7, 10, ... 91, 94]
    # v_slice = [2, 5, 8, 11, ... 92, 95]
    # 70b (with GQA): num_query_groups = 8, head_num = 64
    # q_slice = [0, 1, .. 6, 7, 10, 11, .. 16, 17, 20, 21, .. 67, 70, ... 76, 77]
    # k_slice = [8, 18, 28, ... 68, 78]
    # v_slice = [9, 19, 29, ... 69, 79]

    q_weights_base_name: str = f"model.module.layers.{layer}.self_attn.q_proj.weight"
    k_weights_base_name: str = f"model.module.layers.{layer}.self_attn.k_proj.weight"
    v_weights_base_name: str = f"model.module.layers.{layer}.self_attn.v_proj.weight"

    converted[q_weights_base_name] = param_to_weights(
        qkv_weights[q_slice].reshape(-1, hidden_size)
    )
    converted[k_weights_base_name] = param_to_weights(
        qkv_weights[k_slice].reshape(-1, hidden_size)
    )
    converted[v_weights_base_name] = param_to_weights(
        qkv_weights[v_slice].reshape(-1, hidden_size)
    )

    # attention dense
    o_weight = model.state_dict()[
        f"model.module.decoder.layers.{layer}.self_attention.linear_proj.weight"
    ]
    o_weight_base_name = f"model.module.layers.{layer}.self_attn.o_proj.weight"
    converted[o_weight_base_name] = param_to_weights(o_weight)

    # mlp
    mlp_weights = model.state_dict()[
        f"model.module.decoder.layers.{layer}.mlp.linear_fc1.weight"
    ]
    mlp_down_proj_weight = mlp_weights[:ffn_hidden_size, :]
    mlp_gate_proj_weight = mlp_weights[ffn_hidden_size:, :]

    mlp_down_proj_base_name = f"model.module.layers.{layer}.mlp.gate_proj.weight"
    mlp_gate_proj_base_name = f"model.module.layers.{layer}.mlp.up_proj.weight"

    converted[mlp_down_proj_base_name] = param_to_weights(mlp_down_proj_weight)
    converted[mlp_gate_proj_base_name] = param_to_weights(mlp_gate_proj_weight)

    mlp_up_proj_weight = model.state_dict()[
        f"model.module.decoder.layers.{layer}.mlp.linear_fc2.weight"
    ]
    mlp_up_proj_base_name = f"model.module.layers.{layer}.mlp.down_proj.weight"
    converted[mlp_up_proj_base_name] = param_to_weights(mlp_up_proj_weight)

    # layernorm
    input_ln_weight = model.state_dict()[
        f"model.module.decoder.layers.{layer}.self_attention.linear_qkv.layer_norm_weight"
    ]
    input_ln_base_name = f"model.module.layers.{layer}.input_layernorm.weight"
    converted[input_ln_base_name] = param_to_weights(input_ln_weight)

    post_attn_ln_weight = model.state_dict()[
        f"model.module.decoder.layers.{layer}.mlp.linear_fc1.layer_norm_weight"
    ]
    post_attn_ln_base_name: str = (
        f"model.module.layers.{layer}.post_attention_layernorm.weight"
    )
    converted[post_attn_ln_base_name] = param_to_weights(post_attn_ln_weight)

    logging.info(f"Layer {layer} converted")

    return converted


def convert(
    input_nemo_file: str,
    output_hf_file,
    precision=None,
    cpu_only=False,
    n_jobs=1,
) -> torch.dtype:
    """
    Convert NeMo weights to HF weights
    """
    dummy_trainer = Trainer(devices=1, accelerator="cpu", strategy=NLPDDPStrategy())
    model_config = MegatronGPTModel.restore_from(
        input_nemo_file, trainer=dummy_trainer, return_config=True
    )
    model_config.tensor_model_parallel_size = 1
    model_config.pipeline_model_parallel_size = 1
    model_config.sequence_parallel = False
    if cpu_only:
        logging.info(
            "******** Loading model on CPU. This will take a significant amount of time."
        )
        map_location = torch.device("cpu")
        model_config.use_cpu_initialization = True
    else:
        map_location = None

    model = MegatronGPTModel.restore_from(
        input_nemo_file,
        trainer=dummy_trainer,
        override_config_path=model_config,
        map_location=map_location,
    )
    if precision is None:
        precision = model.cfg.precision
    if precision in [32, "32"]:
        torch_dtype = torch.float32
    elif precision in [16, "16", "16-mixed"]:
        torch_dtype = torch.float16
    elif precision in ["bf16", "bf16-mixed"]:
        torch_dtype = torch.bfloat16
    else:
        logging.warning(
            f"Precision string {precision} is not recognized, falling back to fp32"
        )
        torch_dtype = torch.float32  # fallback
    logging.info(f"Using precision {torch_dtype}")

    checkpoint = OrderedDict()

    param_to_weights = lambda param: param.to(torch_dtype)  # noqa

    for k in model.state_dict().keys():
        logging.info(k)

    # Embedding
    embed_weight = model.state_dict()["model.module.embedding.word_embeddings.weight"]
    checkpoint["model.embed_tokens.weight"] = param_to_weights(embed_weight)

    # Layers
    num_layers: int = model.cfg.num_layers
    n_jobs = min(n_jobs, num_layers)
    converted_layers: list[dict] = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(convert_layer)(model, layer, torch_dtype) for layer in range(num_layers)
    )
    for converted_layer in converted_layers:
        checkpoint.update(converted_layer)

    final_ln_weight = model.state_dict()["model.module.decoder.final_layernorm.weight"]
    checkpoint["model.module.norm.weight"] = param_to_weights(final_ln_weight)
    output_layer_weight = model.state_dict()["model.module.output_layer.weight"]
    checkpoint["lm_head.weight"] = param_to_weights(output_layer_weight)

    if model_config.get("megatron_amp_O2", False):
        keys: list[str] = list(checkpoint.keys())
        for key in keys:
            checkpoint[key.replace("model.module.", "model.", 1)] = checkpoint.pop(key)

    os.makedirs(os.path.dirname(output_hf_file), exist_ok=True)
    torch.save(checkpoint, output_hf_file)
    logging.info(f"Weights saved to {output_hf_file}")

    return torch_dtype


def replace_hf_weights_and_tokenizer(
    weights_file: str,
    torch_dtype: torch.dtype,
    input_hf_path: str,
    output_hf_path: str,
) -> None:
    orig_hf_model = AutoModelForCausalLM.from_pretrained(
        input_hf_path, torch_dtype=torch_dtype
    )
    nemo_weights = torch.load(weights_file)

    tokenizer = AutoTokenizer.from_pretrained(input_hf_path)
    tokenizer_length: int = len(tokenizer)
    if tokenizer_length > orig_hf_model.config.vocab_size:
        logging.warning(
            f"Tokenizer length {tokenizer_length} does not match model vocab size {orig_hf_model.config.vocab_size}. Resizing token embeddings."
        )
        orig_hf_model.resize_token_embeddings(tokenizer_length)

    orig_hf_model.load_state_dict(nemo_weights)
    orig_hf_model.save_pretrained(output_hf_path)
    logging.info(f"Full HF model saved to {output_hf_path}")

    tokenizer.save_pretrained(output_hf_path)
    logging.info(f"Tokenizer saved to {output_hf_path}")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input-name-or-path",
        type=str,
        default=None,
        required=True,
        help="Path to .nemo file or extracted folder",
    )
    parser.add_argument(
        "--input-hf-path",
        type=str,
        default="llm-jp/llm-jp-13b-v2.0",
        help="Path to the HF model directory to use as input",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        required=True,
        help="Output HF model directory",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Precision of output weights. Defaults to precision of the input nemo weights (model.cfg.trainer.precision)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Load model in cpu only. Useful if the model cannot fit in GPU memory, but this option makes the conversion script significantly slower.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs to run. Default is 1. This is for converting the model layers in parallel. If n_jobs > num_layers, it will be set to num_layers.",
    )
    args = parser.parse_args()

    torch_dtype = convert(
        input_nemo_file=args.input_name_or_path,
        output_hf_file=f"{args.output_path}/pytorch_model.bin",
        precision=args.precision,
        cpu_only=args.cpu_only,
        n_jobs=args.n_jobs,
    )
    replace_hf_weights_and_tokenizer(
        weights_file=f"{args.output_path}/pytorch_model.bin",
        torch_dtype=torch_dtype,
        input_hf_path=args.input_hf_path,
        output_hf_path=args.output_path,
    )
    os.remove(f"{args.output_path}/pytorch_model.bin")


if __name__ == "__main__":
    main()
