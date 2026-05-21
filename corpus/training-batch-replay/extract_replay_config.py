#!/usr/bin/env python3

import argparse
import shlex
import subprocess
from pathlib import Path


REPLAY_OPTION_NAMES = {
    "global-batch-size",
    "train-iters",
    "seq-length",
    "seed",
    "split",
    "tokenizer-type",
    "tokenizer-model",
    "make-vocab-size-divisible-by",
    "vocab-extra-ids",
    "data-cache-path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract replay_training_batch.py settings from a training task's "
            "params.sh and train_data.sh."
        )
    )
    parser.add_argument(
        "task_dir",
        type=Path,
        help="Training task directory that contains params.sh and train_data.sh.",
    )
    parser.add_argument(
        "--env-dir",
        type=Path,
        default=None,
        help=(
            "Environment root used by params.sh. Defaults to <experiment>/env "
            "when present, then <experiment>/environment."
        ),
    )
    parser.add_argument(
        "--params-sh",
        type=Path,
        default=None,
        help="Override params.sh path. Defaults to TASK_DIR/params.sh.",
    )
    parser.add_argument(
        "--train-data-sh",
        type=Path,
        default=None,
        help="Override train_data.sh path. Defaults to TASK_DIR/train_data.sh.",
    )
    parser.add_argument(
        "--megatron-path",
        type=Path,
        default=None,
        help="Megatron-LM path to include in outputs. Defaults to ENV_DIR/src/Megatron-LM.",
    )
    parser.add_argument(
        "--format",
        choices=("sh",),
        default="sh",
        help="Output format. Only sh is supported.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write extracted config to this file instead of stdout.",
    )
    return parser.parse_args()


def infer_env_dir(task_dir: Path) -> Path:
    experiment_dir = task_dir.parent.parent
    for name in ("env", "environment"):
        candidate = experiment_dir / name
        if candidate.is_dir():
            return candidate
    return experiment_dir / "env"


def source_task_shell(
    task_dir: Path,
    env_dir: Path,
    params_sh: Path,
    train_data_sh: Path,
) -> tuple[list[str], list[str]]:
    script = f"""
set -euo pipefail
TASK_DIR={shlex.quote(str(task_dir))}
ENV_DIR={shlex.quote(str(env_dir))}
export TASK_DIR ENV_DIR
source {shlex.quote(str(train_data_sh))}
source {shlex.quote(str(params_sh))}
printf '__TRAIN_DATA_PATH__\\0'
printf '%s\\0' "${{TRAIN_DATA_PATH[@]}}"
printf '__ALL_PARAMS__\\0'
printf '%s\\0' "${{ALL_PARAMS[@]}}"
"""
    completed = subprocess.run(
        ["bash", "-c", script],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    parts = completed.stdout.split(b"\0")
    values = [part.decode("utf-8") for part in parts if part]

    try:
        train_marker = values.index("__TRAIN_DATA_PATH__")
        params_marker = values.index("__ALL_PARAMS__")
    except ValueError as exc:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"failed to parse sourced shell output:\n{stderr}") from exc

    train_data_path = values[train_marker + 1 : params_marker]
    all_params = values[params_marker + 1 :]
    return train_data_path, all_params


def option_value(all_params: list[str], name: str) -> str | None:
    option = f"--{name}"
    for index, value in enumerate(all_params):
        if value == option:
            next_index = index + 1
            if next_index >= len(all_params) or all_params[next_index].startswith("--"):
                return None
            return all_params[next_index]
    return None


def extract_data_path(all_params: list[str], train_data_path: list[str]) -> list[str]:
    if "--data-path" not in all_params:
        return train_data_path

    start = all_params.index("--data-path") + 1
    end = start
    while end < len(all_params) and not all_params[end].startswith("--"):
        end += 1
    return all_params[start:end]


def build_config(
    task_dir: Path,
    env_dir: Path,
    megatron_path: Path,
    params_sh: Path,
    train_data_sh: Path,
    train_data_path: list[str],
    all_params: list[str],
) -> dict[str, object]:
    replay_args: dict[str, object] = {}
    for name in sorted(REPLAY_OPTION_NAMES):
        value = option_value(all_params, name)
        if value is not None:
            replay_args[name] = value

    replay_args["data-path"] = extract_data_path(all_params, train_data_path)
    replay_args["megatron-path"] = str(megatron_path)

    return {
        "task_dir": str(task_dir),
        "env_dir": str(env_dir),
        "params_sh": str(params_sh),
        "train_data_sh": str(train_data_sh),
        "replay_args": replay_args,
        "all_params": all_params,
        "train_data_path": train_data_path,
    }


def quote_shell_array(values: list[str]) -> str:
    return " ".join(shlex.quote(value) for value in values)


def iter_sh_lines(config: dict[str, object]):
    args = config["replay_args"]
    if not isinstance(args, dict):
        raise TypeError("replay_args must be a dict")
    yield "# Generated by extract_replay_config.py"
    yield yield_default_assignment("TASK_DIR", str(config["task_dir"]))
    yield yield_default_assignment("ENV_DIR", str(config["env_dir"]))
    mapping = {
        "MEGATRON_PATH": "megatron-path",
        "GLOBAL_BATCH_SIZE": "global-batch-size",
        "TRAIN_STEPS": "train-iters",
        "SEQ_LENGTH": "seq-length",
        "SEED": "seed",
        "SPLIT": "split",
        "TOKENIZER_TYPE": "tokenizer-type",
        "TOKENIZER_MODEL": "tokenizer-model",
        "MAKE_VOCAB_SIZE_DIVISIBLE_BY": "make-vocab-size-divisible-by",
        "VOCAB_EXTRA_IDS": "vocab-extra-ids",
        "DATA_CACHE_PATH": "data-cache-path",
    }
    for env_name, arg_name in mapping.items():
        if arg_name in args:
            yield yield_default_assignment(env_name, str(args[arg_name]))
    if "data-path" in args:
        data_path = args["data-path"]
        if not isinstance(data_path, list):
            raise TypeError("data-path must be a list")
        yield yield_default_assignment("DATA_PATH", quote_shell_array(data_path))


def yield_default_assignment(name: str, value: str) -> str:
    return f"export {name}=${{{name}:-{shlex.quote(value)}}}"


def emit_sh(config: dict[str, object], output_path: Path | None) -> None:
    text = "\n".join(iter_sh_lines(config)) + "\n"
    if output_path is None:
        print(text, end="")
        return
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    task_dir = args.task_dir.expanduser().resolve()
    env_dir = (args.env_dir.expanduser().resolve() if args.env_dir else infer_env_dir(task_dir))
    params_sh = (args.params_sh.expanduser().resolve() if args.params_sh else task_dir / "params.sh")
    train_data_sh = (
        args.train_data_sh.expanduser().resolve()
        if args.train_data_sh
        else task_dir / "train_data.sh"
    )
    megatron_path = (
        args.megatron_path.expanduser().resolve()
        if args.megatron_path
        else env_dir / "src" / "Megatron-LM"
    )

    for path in (task_dir, params_sh, train_data_sh):
        if not path.exists():
            raise FileNotFoundError(path)

    train_data_path, all_params = source_task_shell(
        task_dir=task_dir,
        env_dir=env_dir,
        params_sh=params_sh,
        train_data_sh=train_data_sh,
    )
    config = build_config(
        task_dir=task_dir,
        env_dir=env_dir,
        megatron_path=megatron_path,
        params_sh=params_sh,
        train_data_sh=train_data_sh,
        train_data_path=train_data_path,
        all_params=all_params,
    )

    emit_sh(config, args.output)


if __name__ == "__main__":
    main()
