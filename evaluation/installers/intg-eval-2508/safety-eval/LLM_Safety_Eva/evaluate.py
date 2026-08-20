import json
from pathlib import Path
import yaml
import importlib
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading

# -------------------------------
# 配置
# -------------------------------
CONFIG_PATH = "config.yaml"
MODEL_OUTPUT_DIR = "model_output"
EVALUATOR_OUTPUT_DIR = "evaluator_output"

MAX_WORKERS = 8
RETRIES = 10

# 每次 evaluator.evaluate 请求之间至少间隔 0.5 秒（仅对联网 evaluator 生效）
REQUEST_INTERVAL = 0.5

# 全局限速锁
rate_limit_lock = threading.Lock()
last_request_time = 0.0

Path(EVALUATOR_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# -------------------------------
# 加载配置
# -------------------------------
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

evaluation_items = config.get("evaluation_items", [])
models = config.get("models", [])
benchmark_data = set(config.get("benchmark_data", []))

print("Evaluators to use:", evaluation_items)
print("Models to evaluate:", [m["name"] for m in models])
print("Benchmarks to evaluate:", benchmark_data)

# -------------------------------
# 动态加载 evaluator 类
# -------------------------------
evaluator_classes = {}

for item in evaluation_items:
    module = importlib.import_module(f"evaluators.{item}")
    class_name = item.capitalize() + "Evaluator"
    evaluator_classes[item] = getattr(module, class_name)


# ✅改动：判断某个 evaluator 是否需要联网（默认 True，保持原有 LLM judge 行为）
def evaluator_requires_network(cls):
    return getattr(cls, "requires_network", True)


# 整批评测里，只要有任何一个 evaluator 联网，就需要全局限速。
ANY_NETWORK = any(
    evaluator_requires_network(cls) for cls in evaluator_classes.values()
)
print("Any networked evaluator -> rate limiting:", ANY_NETWORK)


# ==========================================================
# 根据 benchmark 名称选择 eva_prompt
# ==========================================================
def get_eva_prompt_name(config, benchmark_name):
    eva_prompt = config.get("eva_prompt", "V1")

    if isinstance(eva_prompt, dict):
        return eva_prompt.get(benchmark_name, eva_prompt.get("default", "V1"))

    if isinstance(eva_prompt, list):
        return eva_prompt[0] if eva_prompt else "V1"

    return eva_prompt


# ==========================================================
# 读取 benchmark 原始数据，用 prompt_number 补充官方评测字段
# ==========================================================
def load_benchmark_items(base_name):
    benchmark_file = Path("benchmark_data") / f"{base_name}.json"

    if not benchmark_file.exists():
        print(f"⚠️ benchmark 文件不存在: {benchmark_file}")
        return []

    with open(benchmark_file, "r", encoding="utf-8") as f:
        return json.load(f)


def enrich_entry_with_benchmark(entry, benchmark_items):
    """
    把模型输出 entry 和原始 benchmark 数据合并。
    JBBQ 的 label/context_condition/stereotype_label/unk_label 都在 benchmark 原始数据里。
    """
    merged = dict(entry)
    idx = int(entry.get("prompt_number", 1)) - 1

    if 0 <= idx < len(benchmark_items):
        item = benchmark_items[idx]

        if isinstance(item, dict):
            # merged.update(item) 会把 benchmark 原始全部字段并进来，
            # 包含 jbbq 需要的 label / context_condition / stereotype_label / unk_label。
            merged.update(item)
            merged["input"] = item.get("input", entry.get("prompt", ""))
            merged["eval_aspect"] = item.get("eval_aspect", "")
            merged["safety"] = item.get("safety", "")
            merged["ng_aspect"] = item.get("ng_aspect", "")
            merged["type"] = item.get("type", "")
            merged["category"] = item.get("category", "")
        else:
            merged["input"] = str(item)
            merged["eval_aspect"] = ""
            merged["safety"] = ""
            merged["ng_aspect"] = ""
    else:
        merged["input"] = entry.get("prompt", "")
        merged["eval_aspect"] = ""
        merged["safety"] = ""
        merged["ng_aspect"] = ""

    merged["lm_output"] = entry.get("output", "")
    merged["prompt"] = merged.get("prompt") or merged.get("input", "")

    return merged


# ==========================================================
# 全局请求限速（仅联网 evaluator 调用）
# ==========================================================
def wait_for_rate_limit():
    global last_request_time

    with rate_limit_lock:
        now = time.time()
        elapsed = now - last_request_time

        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)

        last_request_time = time.time()


# ==========================================================
# 带重试逻辑
# ==========================================================
# ✅改动：新增 needs_network 参数。不联网的 evaluator 不限速、失败也不空等。
def evaluate_with_retry(evaluator, question, output, key_name, entry=None,
                        needs_network=True):
    for attempt in range(1, RETRIES + 1):
        try:
            # ✅改动：只有联网 evaluator 才走全局限速
            if needs_network:
                wait_for_rate_limit()

            result = evaluator.evaluate(question, output, entry=entry)

            if result is not None:
                # 纯本地评测每条都成功，日志太吵，这里只在联网时打成功日志
                if needs_network:
                    print(f"✔ {key_name} 成功: 尝试 {attempt}/{RETRIES}")
                return result

            print(f"⚠️ {key_name} 返回 None（第 {attempt} 次）")

        except Exception as e:
            print(f"❌ {key_name} 异常（第 {attempt} 次）: {e}")

        # ✅改动：联网才需要失败后等待退避；本地纯函数没必要 sleep
        if needs_network:
            time.sleep(0.5)

    print(f"⚠️ {key_name} 评估 {RETRIES} 次全部失败 → 写入 null")
    return None


# ==========================================================
# 评估单条 entry
# ==========================================================
def evaluate_single(entry, evaluator_classes, benchmark_name, config):
    result = {
        "prompt_number": entry.get("prompt_number"),
        "prompt": entry.get("prompt", ""),
        "attempt": entry.get("attempt"),
        "output": entry.get("output", ""),
    }

    # ✅改动：保留官方字段 + JBBQ 聚合必需字段。
    # label / context_condition / stereotype_label / streotype_label / unk_label
    # 不保留的话，jbbq.summarize 读不到会算不出分。
    for k in [
        "type", "category", "safety", "input", "eval_aspect", "ng_aspect",
        "label", "context_condition", "stereotype_label", "streotype_label",
        "unk_label",
    ]:
        if k in entry:
            result[k] = entry.get(k, "")

    prompt_name = get_eva_prompt_name(config, benchmark_name)

    for key, cls in evaluator_classes.items():
        evaluator = cls(prompt_name=prompt_name)
        needs_network = evaluator_requires_network(cls)   # ✅改动

        question = entry.get("input", entry.get("prompt", ""))
        output = entry.get("output", "")

        res = evaluate_with_retry(
            evaluator=evaluator,
            question=question,
            output=output,
            key_name=f"{key}:{prompt_name}",
            entry=entry,
            needs_network=needs_network,   # ✅改动
        )

        result[key] = res

    return result


# ==========================================================
# 处理 JSON 文件：normal / think / base
# ==========================================================
def get_json_paths(model_folder, base_name):
    paths = {}

    normal = model_folder / f"{base_name}_normal.json"
    think = model_folder / f"{base_name}_think.json"
    base = model_folder / f"{base_name}.json"

    if normal.exists():
        paths["normal"] = normal

    if think.exists():
        paths["think"] = think

    if not paths and base.exists():
        paths["base"] = base

    return paths


# ==========================================================
# 主流程
# ==========================================================
for model in models:
    model_name = model["name"].split("/")[-1]
    model_folder = Path(MODEL_OUTPUT_DIR) / f"{model_name}_output"

    if not model_folder.exists():
        print(f"⚠️ 模型文件夹 {model_folder} 不存在，跳过。")
        continue

    print(f"\n📂 Processing model folder: {model_folder.name}")

    output_folder = Path(EVALUATOR_OUTPUT_DIR) / f"{model_name}_evaluate"
    output_folder.mkdir(parents=True, exist_ok=True)

    all_files = list(model_folder.glob("*.json"))

    if not all_files:
        print(f"⚠️ {model_folder} 无 JSON 文件，跳过")
        continue

    all_base_names = set(
        f.stem.replace("_normal", "").replace("_think", "")
        for f in all_files
    )

    base_names = all_base_names.intersection(benchmark_data)

    if not base_names:
        print("⚠️ 无匹配 benchmark_data 的文件，跳过")
        print("已有输出文件:", sorted(all_base_names))
        print("config benchmark_data:", sorted(benchmark_data))
        continue

    for base_name in base_names:
        json_paths = get_json_paths(model_folder, base_name)

        if not json_paths:
            print(f"⚠️ 找不到 {base_name} 的 JSON 文件")
            continue

        benchmark_items = load_benchmark_items(base_name)

        for mode, json_file in json_paths.items():
            print(f"\n🧩 Evaluating {json_file.name} ...")
            print(f"🧾 Prompt template: {get_eva_prompt_name(config, base_name)}")

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 调试时可只评前 5 条
            # data = data[:5]

            enriched_data = [
                enrich_entry_with_benchmark(entry, benchmark_items)
                for entry in data
            ]

            results = []

            # ✅改动：本地评测不需要多线程（也不需要限速）；
            # 用更多 worker 加速即可，无 IO 阻塞反而线程切换有点开销，但保持结构一致。
            workers = MAX_WORKERS

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        evaluate_single,
                        entry,
                        evaluator_classes,
                        base_name,
                        config,
                    )
                    for entry in enriched_data
                ]

                for fut in as_completed(futures):
                    results.append(fut.result())

            out_file = output_folder / f"{base_name}_{mode}_evaluated_{datetime.date.today()}.json"

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"✅ 已保存: {out_file}")

            # summary 不在这一步生成；统计交给 evaluate_count.py 处理。