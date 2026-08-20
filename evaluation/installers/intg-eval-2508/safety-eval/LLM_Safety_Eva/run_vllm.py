# ============================================================
# run_vllm.py
# ============================================================
# 一个脚本连续跑多个 vLLM 模型版本
# 重点：
# - 环境变量必须在所有 torch / transformers / vllm import 前设置
# - 不主动 destroy torch.distributed / vLLM distributed state
# - 模型结束后先真正 del llm，再等待旧 multiprocessing.spawn worker 消失
# ============================================================

import os

# ============================================================
# 必须最先设置
# ============================================================
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

# 你的报错/卡住点在 gloo/c10d 初始化，所以优先固定 Gloo 用本机 loopback
os.environ["GLOO_SOCKET_IFNAME"] = "lo"

# 先不要开 NCCL_SOCKET_IFNAME=lo；如果仍然卡住，再试着取消下面这一行注释
# os.environ["NCCL_SOCKET_IFNAME"] = "lo"

import yaml
import json
import shutil
import gc
import time
import subprocess
import multiprocessing as mp

from tqdm import tqdm
from transformers import set_seed, AutoTokenizer
from vllm import LLM, SamplingParams


set_seed(1234)


# ============================================================
# 配置读取
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
# vLLM 加载
# ============================================================
def load_local_vllm(model_name, tensor_parallel_size=8):
    print("\n=======================================", flush=True)
    print(
        f"🔄 Loading model by vLLM ({tensor_parallel_size} GPUs): {model_name}",
        flush=True,
    )
    print("=======================================\n", flush=True)

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_num_seqs=32,
        max_model_len=4096,

        # 为了连续加载多个模型时稳定
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )

    return llm


# ============================================================
# prompt / batch
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
# 等待 vLLM worker 真正退出
# ============================================================
def list_vllm_worker_processes():
    cmd = (
        "ps -u $USER -f | "
        "grep -E 'multiprocessing.spawn|VllmWorker|VllmWorkerProcess|multiproc_worker' | "
        "grep -v grep | "
        "grep -v jupyter"
    )

    result = subprocess.run(
        ["bash", "-lc", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return result.stdout.strip()


def wait_until_vllm_workers_gone(timeout=240):
    start = time.time()

    while time.time() - start < timeout:
        workers = list_vllm_worker_processes()

        if workers == "":
            print("[cleanup] no vLLM/multiprocessing worker remains", flush=True)
            return True

        print("[cleanup] waiting for old vLLM workers to exit ...", flush=True)
        print(workers, flush=True)
        time.sleep(5)

    print("[cleanup] timeout waiting for workers; killing remaining workers ...", flush=True)

    kill_cmds = [
        "pkill -u $USER -f 'multiprocessing.spawn'",
        "pkill -u $USER -f 'VllmWorker'",
        "pkill -u $USER -f 'VllmWorkerProcess'",
    ]

    for cmd in kill_cmds:
        subprocess.run(
            ["bash", "-lc", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    time.sleep(15)

    workers = list_vllm_worker_processes()
    if workers:
        print("[cleanup] workers still remain after pkill:", flush=True)
        print(workers, flush=True)
        return False

    print("[cleanup] workers killed successfully", flush=True)
    return True


def cleanup_after_vllm():
    print("\n🧹 cleanup_after_vllm() started ...", flush=True)

    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("[cleanup] torch cuda cache cleared", flush=True)

    except Exception as e:
        print(f"[cleanup] torch cuda cleanup skipped: {e}", flush=True)

    gc.collect()

    wait_until_vllm_workers_gone(timeout=240)

    print("[cleanup] extra sleep 30s for ports/gloo/nccl release ...", flush=True)
    time.sleep(30)

    print("🧹 cleanup_after_vllm() finished.\n", flush=True)


# ============================================================
# 单个 benchmark
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
    config = load_config()

    models = config.get("models", [])
    benchmarks = config.get("benchmark_data", [])
    ask_times = int(config.get("ask_times", 1))

    tensor_parallel_size = int(config.get("tensor_parallel_size", 8))
    batch_size = int(config.get("batch_size", 32))

    max_tokens = int(config.get("max_tokens", 1024))
    temperature = float(config.get("temperature", 1.0))
    top_p = float(config.get("top_p", 0.95))

    print("Benchmarks:", benchmarks, flush=True)
    print("Models:", [m["name"] for m in models], flush=True)
    print("tensor_parallel_size:", tensor_parallel_size, flush=True)
    print("batch_size:", batch_size, flush=True)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # 启动前确认没有旧 worker
    existing_workers = list_vllm_worker_processes()
    if existing_workers:
        print("[startup] old workers detected before run:", flush=True)
        print(existing_workers, flush=True)
        print("[startup] waiting/killing old workers first ...", flush=True)
        wait_until_vllm_workers_gone(timeout=30)

    for model in models:
        model_full = model["name"]
        model_name = model_full.split("/")[-1]

        tokenizer = None
        llm = None

        try:
            tokenizer = load_tokenizer(model_full)

            llm = load_local_vllm(
                model_full,
                tensor_parallel_size=tensor_parallel_size,
            )

            model_output_dir = f"./model_output/{model_name}_output"

            if os.path.exists(model_output_dir):
                shutil.rmtree(model_output_dir)

            os.makedirs(model_output_dir, exist_ok=True)

            for benchmark in benchmarks:
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
            print(f"\n❌ Error while running model: {model_full}", flush=True)
            print(f"❌ Exception: {repr(e)}", flush=True)
            raise

        finally:
            print(f"\n🧹 Releasing GPU memory for {model_full} ...\n", flush=True)

            # 关键：必须在 cleanup 前真正删除外层 llm 引用
            try:
                if tokenizer is not None:
                    del tokenizer
                tokenizer = None
            except Exception:
                pass

            try:
                if llm is not None:
                    del llm
                llm = None
                print("[cleanup] outer llm reference deleted", flush=True)
            except Exception as e:
                print(f"[cleanup] deleting outer llm failed: {e}", flush=True)

            cleanup_after_vllm()

    print("\n🎉🎉 All models finished (batch mode)!\n", flush=True)


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    mp.freeze_support()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main()