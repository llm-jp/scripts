# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, LLM-jp.
import sys
import os
import shutil
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, LlamaConfig
from contextlib import contextmanager

def add_arguments(parser):
    group = parser.add_argument_group(title="HF saver (LLM-jp style)")
    group.add_argument(
        "--hf-tokenizer-path",
        type=str,
        default=None,
        help="Path to an existing HF tokenizer directory. All files will be copied into save-dir.",
    )
    group.add_argument(
        "--save-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32", "float16"],
    )

@contextmanager
def suspend_nn_inits():
    def noop(*args, **kwargs):
        return None
    orig_kaiming_uniform = torch.nn.init.kaiming_uniform_
    orig_uniform = torch.nn.init.uniform_
    orig_normal = torch.nn.init.normal_
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_ = orig_kaiming_uniform
        torch.nn.init.uniform_ = orig_uniform
        torch.nn.init.normal_ = orig_normal

def save_checkpoint(queue: mp.Queue, args):
    # Make sure we can import Megatron
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    def queue_get(name=None):
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and args.checking and val["name"] != name:
            print(f"Unexpected message. Expecting '{name}' but got '{val['name']}'. Exiting saver.")
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    def check_message(msg):
        if not args.checking:
            return
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            print(f"Unexpected values in {msg_name}: {list(msg.keys())}")
            print("Exiting. If you want to ignore this, use --no-checking.")
            exit(1)

    # Metadata
    md = queue_get()
    assert hasattr(md, "checkpoint_args") and md.model_type == "GPT"
    m = md.checkpoint_args

    # Dtype
    torch_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}[args.save_dtype]

    # HF config
    llama_conf = LlamaConfig(
        vocab_size=m.padded_vocab_size,
        hidden_size=m.hidden_size,
        intermediate_size=m.ffn_hidden_size,
        num_hidden_layers=m.encoder_num_layers,
        num_attention_heads=m.num_attention_heads,
        num_key_value_heads=m.num_query_groups if m.group_query_attention else m.num_attention_heads,
        max_position_embeddings=m.max_position_embeddings,
        rms_norm_eps=m.norm_epsilon,
        tie_word_embeddings=not m.untie_embeddings_and_output_weights,
        rope_theta=m.rotary_base,
        attention_bias=m.add_bias_linear,
        torch_dtype=torch_dtype,
    )

    # Collect weights from queue
    state_dict = {}
    def set_w(name, tensor: torch.Tensor):
        state_dict[f"{name}.weight"] = tensor.to(torch.bfloat16)

    emb = queue_get("embeddings")
    set_w("model.embed_tokens", emb["word embeddings"])

    for i in range(llama_conf.num_hidden_layers):
        msg = queue_get(f"transformer layer {i}")
        p = f"model.layers.{i}"
        set_w(f"{p}.input_layernorm", msg["input norm weight"])
        set_w(f"{p}.post_attention_layernorm", msg["post norm weight"])
        set_w(f"{p}.mlp.gate_proj", msg["mlp l0 weight W"])
        set_w(f"{p}.mlp.up_proj", msg["mlp l0 weight V"])

        qkv = msg["qkv weight"]
        qkv = qkv.view(llama_conf.num_key_value_heads, -1, llama_conf.hidden_size)
        qkv = torch.split(qkv, [
            llama_conf.hidden_size // llama_conf.num_key_value_heads,
            llama_conf.hidden_size // llama_conf.num_attention_heads,
            llama_conf.hidden_size // llama_conf.num_attention_heads,
        ], dim=1)
        set_w(f"{p}.self_attn.q_proj", qkv[0].reshape(-1, llama_conf.hidden_size))
        set_w(f"{p}.self_attn.k_proj", qkv[1].reshape(-1, llama_conf.hidden_size))
        set_w(f"{p}.self_attn.v_proj", qkv[2].reshape(-1, llama_conf.hidden_size))
        set_w(f"{p}.self_attn.o_proj", msg["dense weight"])
        set_w(f"{p}.mlp.down_proj", msg["mlp l1 weight"])

    fn = queue_get("final norm")
    set_w("model.norm", fn["weight"])

    out = queue_get("output layer")
    set_w("lm_head", out["weight"])

    # 5) Build HF model and save
    os.makedirs(args.save_dir, exist_ok=True)
    with suspend_nn_inits():
        print("Saving model to disk ...")
        model = AutoModelForCausalLM.from_config(llama_conf, torch_dtype=torch_dtype)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"Warning: load_state_dict mismatches. missing={len(missing)} unexpected={len(unexpected)}")
        model.save_pretrained(args.save_dir, safe_serialization=True)

    # 6) Copy HF tokenizer files as-is
    if args.hf_tokenizer_path is not None and os.path.isdir(args.hf_tokenizer_path):
        for entry in os.listdir(args.hf_tokenizer_path):
            s = os.path.join(args.hf_tokenizer_path, entry)
            d = os.path.join(args.save_dir, entry)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
    else:
        print("NOTE: --hf-tokenizer-path not provided or not a directory; skipping tokenizer copy.")