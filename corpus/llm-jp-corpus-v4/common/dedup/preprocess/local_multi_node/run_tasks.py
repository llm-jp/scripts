import getpass
from concurrent.futures import ThreadPoolExecutor, as_completed

import paramiko
from tqdm import tqdm

# === 設定項目 ===
servers = [f"z-cpu{i}" for i in range(63)]
username = getpass.getuser()
target_dir = "/model/experiments/0118_dedup_corpusv4_ja"
python_script = "scripts/corpus/llm-jp-corpus-v4/common/dedup/propcess/local_multi_node/minhash_dedup.py"
task_per_server=100
python_args=[
    "data/all/deduped_subcorpus",
    "data/all",
    "--local_tasks",
    f"{task_per_server}",
    "--local_rank_offset",
]
log_dir = "local_logs/all-stage1"
max_workers = 10
# ===========================


def run_task(server, task_number):
    try:
        # SSHクライアントの準備
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(
            paramiko.AutoAddPolicy()
        )  # ホストキーの確認を無効化

        # SSHエージェントから鍵を取得
        agent = paramiko.Agent()
        keys = agent.get_keys()
        if len(keys) == 0:
            raise Exception(
                "SSHエージェントに鍵が登録されていません。`ssh-add`で鍵を追加してください。"
            )

        # 実行ユーザー名でSSH接続
        ssh.connect(server, username=username, pkey=keys[0])  # エージェントの鍵を使用

        # 実行コマンド
        args_concat=" ".join(python_args)
        args_concat+=f" {task_per_server*task_number}"
        
        python_commands=f"python {python_script} {args_concat}"
        python_logging=f"> {log_dir}/{server}.out 2>&1"
        
        command = f"""
            cd {target_dir} && \
            source environment/.venv/bin/activate && \
            nohup bash -c '{python_commands} {python_logging}' > nohup.out 2>&1 &
        """
        #command = "pkill -u ytsuta python"
        ssh.exec_command(command)
        ssh.close()

        return f"{server}: ✅ Task Started (Task #{task_number}) as {username}"
    except Exception as e:
        return f"{server}: ❌ FAILED - {str(e)}"


# 並列実行と進捗バー表示
if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_task, server, i) for i, server in enumerate(servers)
        ]
        for future in tqdm(
            as_completed(futures), total=len(servers), desc="Running Tasks"
        ):
            print(future.result())
