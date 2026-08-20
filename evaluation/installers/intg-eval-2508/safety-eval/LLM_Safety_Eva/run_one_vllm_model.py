import os
import sys

# ============================================================
# 必须在 import torch / transformers / vllm 之前设置
# ============================================================
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# 多卡 NCCL 保守模式：牺牲性能，换稳定性
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"

import argparse
import yaml
import json
import gc

from tqdm import tqdm
from transformers import set_seed, AutoTokenizer
from vllm import LLM, SamplingParams


set_seed(1234)


# ============================================================
# 读取配置
# ============================================================
def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# Tokenizer
# ============================================================
def load_tokenizer(model_name):
    print(f"\n🔄 Loading tokenizer: {model_name}\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    print(
        "tokenizer.chat_template exists?:",
        getattr(tokenizer, "chat_template", None) is not None,
        flush=True,
    )

    return tokenizer


def format_prompt(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]

    fallback_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "<|system|>\n{{ message['content'] }}\n"
        "{% elif message['role'] == 'user' %}"
        "<|user|>\n{{ message['content'] }}\n"
        "{% elif message['role'] == 'assistant' %}"
        "<|assistant|>\n{{ message['content'] }}\n"
        "{% endif %}"
        "{% endfor %}"
        "<|assistant|>\n"
    )

    if getattr(tokenizer, "chat_template", None) is None:
        return tokenizer.apply_chat_template(
            messages,
            chat_template=fallback_template,
            tokenize=False,
            add_generation_prompt=True,
        )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ============================================================
# prompt 处理
# ============================================================
def get_prompt_text(item):
    if isinstance(item, dict):
        return item.get("input", item.get("prompt", ""))
    return item


def build_batches(input_prompts, ask_times):
    queue = []

    for prompt_idx, item in enumerate(input_prompts, start=1):
        prompt = get_prompt_text(item)

        for attempt in range(ask_times):
            queue.append((prompt_idx, attempt + 1, prompt))

    return queue


# ============================================================
# vLLM 加载
# ============================================================
def load_local_vllm(model_name, tensor_parallel_size):
    print("\n=======================================", flush=True)
    print(
        f"🔄 Loading model by vLLM ({tensor_parallel_size} GPUs): {model_name}",
        flush=True,
    )
    print("=======================================\n", flush=True)

    return LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_num_seqs=128,
        max_model_len=4096,
        enforce_eager=False,
        disable_custom_all_reduce=False,
        # tokenizer_mode="slow",
    )


# ============================================================
# 跑一个 benchmark
# ============================================================
def run_one_benchmark(
    llm,
    tokenizer,
    benchmark,
    ask_times,
    batch_size,
    sampling_params,
    model_output_dir,
):
    benchmark_file = f"benchmark_data/{benchmark}.json"

    with open(benchmark_file, "r", encoding="utf-8") as f:
        input_prompts = json.load(f)

    print(f"\n🚀 Start Benchmark (batch mode): {benchmark}\n", flush=True)

    queue = build_batches(input_prompts, ask_times)
    all_results = []

    for i in tqdm(
        range(0, len(queue), batch_size),
        desc=f"Processing {benchmark}",
    ):
        batch = queue[i:i + batch_size]

        batch_prompts = [
            format_prompt(tokenizer, x[2])
            for x in batch
        ]

        outputs = llm.generate(
            batch_prompts,
            sampling_params=sampling_params,
        )

        for meta, out in zip(batch, outputs):
            prompt_idx, attempt, prompt_text = meta
            ans = out.outputs[0].text

            all_results.append({
                "prompt_number": prompt_idx,
                "prompt": prompt_text,
                "attempt": attempt,
                "output": ans,
            })

    out_path = f"{model_output_dir}/{benchmark}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Saved: {out_path}", flush=True)


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    config = load_config()

    model_full = args.model
    model_name = model_full.split("/")[-1]

    benchmarks = config.get("benchmark_data", [])
    ask_times = int(config.get("ask_times", 1))

    tensor_parallel_size = int(config.get("tensor_parallel_size", 8))
    batch_size = int(config.get("batch_size", 32))

    max_tokens = int(config.get("max_tokens", 1024))
    temperature = float(config.get("temperature", 1.0))
    top_p = float(config.get("top_p", 0.95))

    print("Model:", model_full, flush=True)
    print("Benchmarks:", benchmarks, flush=True)
    print("tensor_parallel_size:", tensor_parallel_size, flush=True)
    print("batch_size:", batch_size, flush=True)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # 用退出码标记成功/失败，交给 os._exit 使用
    exit_code = 0

    tokenizer = None
    llm = None

    try:
        tokenizer = load_tokenizer(model_full)

        llm = load_local_vllm(
            model_full,
            tensor_parallel_size=tensor_parallel_size,
        )

        model_output_dir = f"./model_output/{model_name}_output"
        os.makedirs(model_output_dir, exist_ok=True)

        for benchmark in benchmarks:
            out_path = f"{model_output_dir}/{benchmark}.json"
            if os.path.exists(out_path):
                print(f"  ⏭️  已存在,跳过: {out_path}", flush=True)
                continue

            run_one_benchmark(
                llm=llm,
                tokenizer=tokenizer,
                benchmark=benchmark,
                ask_times=ask_times,
                batch_size=batch_size,
                sampling_params=sampling_params,
                model_output_dir=model_output_dir,
            )

        print(f"\n🎉 Model {model_full} finished.\n", flush=True)

    except Exception as e:
        exit_code = 1
        import traceback
        print(f"\n❌ Error running model {model_full}: {e}", flush=True)
        traceback.print_exc()

    finally:
        # 不做 del llm 的软清理：vLLM V1 + 多卡(TP=8)在销毁引擎时
        # 经常卡在 ZMQ / NCCL 优雅关闭上。这里直接硬退出，
        # 由操作系统回收整个进程(CUDA context + 全部显存 + worker 子进程)，
        # 比 del llm + empty_cache 更干净更可靠。
        print(f"\n🧹 Process exiting (OS will reclaim GPU memory): {model_full}\n", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()