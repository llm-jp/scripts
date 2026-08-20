from utils import storage
import yaml
import json
import requests
import os
import shutil
from transformers import set_seed

set_seed(1234)
# -------------------------------
# 获取配置文件
# -------------------------------
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

models = config.get('models', [])
generate_url = config.get('generate_url', '').strip()
benchmarks = config.get('benchmark_data', [])
load_model_url = config.get('load_model_url', '').strip()
ask_times = config.get('ask_times', '')

print("Benchmarks:", benchmarks)

# -------------------------------
# 遍历模型
# -------------------------------
for model in models:
    model_name = model['name'].split('/')[-1]

    try:
        load_resp = requests.post(load_model_url, json={"model_name": model['name']})
        load_resp.raise_for_status()
        load_json = load_resp.json()
        print("✅ 加载成功:", load_json)
        has_think = load_json.get("has_think", False)  # ✅ 读取是否支持think
    except Exception as e:
        print("❌ 模型加载失败:", e)
        try:
            print("服务器返回:", load_resp.json())
        except:
            print("服务器返回:", load_resp.text)
        print("异常信息:", str(e))
        continue

    model_output_dir = f'./model_output/{model_name}_output'
    if os.path.exists(model_output_dir):
        shutil.rmtree(model_output_dir)
    os.makedirs(model_output_dir, exist_ok=True)

    # -------------------------------
    # 遍历 benchmark
    # -------------------------------
    for benchmark in benchmarks:
        benchmark_file = f'benchmark_data/{benchmark}.json'
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            input_prompts = json.load(f)

        # -------------------------------
        # 遍历 prompt（带编号）
        # -------------------------------
        for prompt_idx, input_prompt in enumerate(input_prompts, start=1):
            think_modes = [False, True] if has_think else [False]  # ✅ 自动判断
            for think_mode in think_modes:
                # ✅ 每个 prompt 生成三次回答
                for i in range(ask_times):
                    data = {
                        'input_prompt': input_prompt,
                        'max_new_tokens': model['max_new_tokens'],
                        'temperature': model['temperature'],
                        'top_p': model['top_p'],
                        'think': think_mode
                    }

                    output = requests.post(generate_url, json=data).json()
                    result = {
                        "prompt_number": prompt_idx,   # ✅ 第几个 prompt
                        "prompt": input_prompt,
                        "think_mode": think_mode,
                        "attempt": i + 1,              # ✅ 第几次生成
                        "output": output.get("generated_text", "")
                    }

                    suffix = "_think" if think_mode else "_normal"
                    storage.append_result_to_json(
                        result=result,
                        folder_path=model_output_dir,
                        json_name=f"{benchmark}{suffix}.json",
                        reset=False
                    )
                    print(f"✅ 已生成: {benchmark}{suffix}.json [Prompt {prompt_idx} | 第{i+1}次]")
