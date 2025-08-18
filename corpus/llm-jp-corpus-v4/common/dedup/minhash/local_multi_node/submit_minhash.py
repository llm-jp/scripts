import argparse
import getpass
import logging
from pathlib import Path

import paramiko

# Args default
WORK_DIR_DEFAULT = "/model/experiments/0118_dedup_corpusv4_ja/data"
PYTHON_SCRIPT_PATH_DEFAULT = (
    WORK_DIR_DEFAULT
    + "/scripts/corpus/llm-jp-corpus-v4/common/dedup/minhash/local_multi_node/minhash_dedup.py"
)
VENV_PATH_DEFAULT = WORK_DIR_DEFAULT + "/environment/.venv/bin/activate"

# server settings
USER_NAME = getpass.getuser()

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Args:
    # For runner
    input_dir: str
    output_dir: str
    stage: int

    # About paths
    log_dir: str
    venv_path: str
    python_script: str

    # About server
    node_list: list[str]
    max_node_worker: int


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    runner_parser = parser.add_argument_group("Arguments for running scripts")
    runner_parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
    )
    runner_parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
    )
    runner_parser.add_argument(
        "-stage", "--stage", type=int, choices=[1, 2, 3, 4], default=4
    )
    path_parser = parser.add_argument_group("Arguments about paths")
    path_parser.add_argument(
        "--log_dir",
        required=True,
        type=str,
    )
    path_parser.add_argument(
        "--venv_path",
        default=VENV_PATH_DEFAULT,
        type=str,
    )
    path_parser.add_argument(
        "--python_script",
        default=PYTHON_SCRIPT_PATH_DEFAULT,
        type=str,
    )
    server_parser = parser.add_argument_group("Arguments about server")
    server_parser.add_argument(
        "--node_list",
        nargs="+",
        type=str,
    )
    server_parser.add_argument(
        "--max_node_worker",
        type=int,
    )

    return parser


# command = "kill -9 -- -1"


def submit_task(node, command):
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
        ssh.connect(node, username=USER_NAME, pkey=keys[0])  # エージェントの鍵を使用

        ssh.exec_command(command)
        ssh.close()

        logger.info(f"{node}: ✅ Task Started as {USER_NAME}")
    except Exception as e:
        logger.info(f"{node}: ❌ FAILED - {str(e)}")


def prepare_command_prefix(
    input_dir: Path,
    output_dir: Path,
    venv_path: Path,
    python_script: Path,
):
    python_args = [input_dir, output_dir]
    concat_args = " ".join(python_args)
    python_commands = f"python {python_script} {concat_args}"
    command_prefix = "&&".join(
        [
            "ulimit -n 65536 1048576",
            f"source {venv_path}",
            f"nohup bash -c '{python_commands}",
        ]
    )
    return command_prefix


def submit_minhash_job(
    node_list: list[str],
    input_dir: Path,
    command_prefix,
    stage: int,
    log_dir: Path,
    max_node_worker: int,
):
    all_files = [_f for _f in input_dir.rglob("*") if _f.resolve().is_file()]
    total_tasks = len(all_files)

    for i, finish_tasks in enumerate(range(0, total_tasks, max_node_worker)):
        rest_tasks = total_tasks - finish_tasks
        local_tasks = min(rest_tasks, max_node_worker)
        node = node_list[i]

        # complete commannd
        log_path = log_dir / (node + ".out")
        logging_command = f"> {log_path} 2>&1"
        nohup_sufix = "> nohup.out 2>&1 &"
        command = "".join(
            [
                command_prefix,
                "--stage", 
                f"{stage}",
                "--local_tasks",
                f"{local_tasks}",
                "--local_rank_offset",
                f"{finish_tasks}",
                f"{logging_command}",
                f"{nohup_sufix}",
            ]
        )

        submit_task(node, command)

        if stage in [2,3]:
            # On stage2 and stage3, process is not distributed
            break


def main(
    input_dir: str,
    output_dir: str,
    stage: int,
    log_dir: str,
    venv_path: str,
    python_script: str,
    node_list: list[str],
    max_node_worker: int,
):
    command_prefix = prepare_command_prefix(
        input_dir, output_dir, venv_path, python_script
    )
    submit_minhash_job(node_list, input_dir, command_prefix,stage, log_dir, max_node_worker)


if __name__ == "__main__":
    args = setup_parser().parse_args(namespace=Args)
    main(
        args.input_dir,
        args.output_dir,
        args.stage,
        args.log_dir,
        args.venv_path,
        args.python_script,
        args.node_list,
        args.max_node_worker,
    )
