import yaml
import subprocess
import time
import os
import sys


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cleanup_workers():
    """
    清理当前用户残留的 vLLM 相关进程。

    vLLM 不同版本的 worker 名字不一样：
    - 旧版可能叫 VllmWorker / VllmWorkerProcess
    - 新版可能叫 VLLM::Worker_TP0 / EngineCore_DP0
    所以这里都匹配一下。
    """

    soft_cmds = [
        "pkill -u $USER -f 'run_one_vllm_model.py' || true",
        "pkill -u $USER -f 'multiprocessing.spawn' || true",
        "pkill -u $USER -f 'VLLM::Worker' || true",
        "pkill -u $USER -f 'VllmWorker' || true",
        "pkill -u $USER -f 'VllmWorkerProcess' || true",
        "pkill -u $USER -f 'EngineCore' || true",
    ]

    for cmd in soft_cmds:
        subprocess.run(["bash", "-lc", cmd], check=False)

    time.sleep(10)

    force_cmds = [
        "pkill -9 -u $USER -f 'run_one_vllm_model.py' || true",
        "pkill -9 -u $USER -f 'multiprocessing.spawn' || true",
        "pkill -9 -u $USER -f 'VLLM::Worker' || true",
        "pkill -9 -u $USER -f 'VllmWorker' || true",
        "pkill -9 -u $USER -f 'VllmWorkerProcess' || true",
        "pkill -9 -u $USER -f 'EngineCore' || true",
    ]

    for cmd in force_cmds:
        subprocess.run(["bash", "-lc", cmd], check=False)

    time.sleep(10)


def show_gpu():
    subprocess.run(["bash", "-lc", "nvidia-smi"], check=False)


def show_leftover_processes():
    cmd = (
        "pgrep -af "
        "'run_one_vllm_model.py|multiprocessing.spawn|VLLM::Worker|"
        "VllmWorker|VllmWorkerProcess|EngineCore' "
        "|| true"
    )

    print("\n🔍 Remaining vLLM-related processes:", flush=True)
    subprocess.run(["bash", "-lc", cmd], check=False)


def main():
    config = load_config()
    models = config.get("models", [])

    print("Python:", sys.executable, flush=True)
    print("Models:", [m["name"] for m in models], flush=True)

    for idx, model in enumerate(models, start=1):
        model_name = model["name"]

        print("\n" + "=" * 80, flush=True)
        print(f"🚀 [{idx}/{len(models)}] Start model: {model_name}", flush=True)
        print("=" * 80 + "\n", flush=True)

        cleanup_workers()
        show_leftover_processes()
        show_gpu()

        cmd = [
            sys.executable,
            "run_one_vllm_model.py",
            "--model",
            model_name,
        ]

        result = subprocess.run(cmd)

        cleanup_workers()
        show_leftover_processes()
        show_gpu()

        if result.returncode != 0:
            print(f"\n❌ Model failed: {model_name}", flush=True)
            print(f"❌ returncode: {result.returncode}", flush=True)
            raise SystemExit(result.returncode)

        print(f"\n✅ Model completed: {model_name}", flush=True)

        print("\n⏳ Sleep 10s before next model ...\n", flush=True)
        time.sleep(10)

    print("\n🎉🎉 All models finished!\n", flush=True)


if __name__ == "__main__":
    main()