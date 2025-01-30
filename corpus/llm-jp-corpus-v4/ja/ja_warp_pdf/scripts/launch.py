import argparse
import logging
from pathlib import Path
import subprocess
from queue import Queue
from time import sleep


logger = logging.getLogger(__name__)

here = Path(__file__).parent

script_paths = {
    "convert": here / "convert.py",
    "filter": here / "filter.py",
}

interpreter_path = here.parent / ".venv" / "bin" / "python"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "MODE",
        type=str,
        choices=["test", "convert", "filter"],
        help="Mode to run",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the input file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output file",
    )
    parser.add_argument(
        "--hosts",
        type=str,
        required=True,
        help="Path to the hosts file",
    )
    args = parser.parse_args()

    script_path = script_paths[args.MODE]

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    queue = Queue()
    for input_path in input_dir.glob("**/*.jsonl"):
        output_path = output_dir / input_path.relative_to(input_dir)
        command = [
            str(interpreter_path),
            str(script_path),
            "--input-file",
            str(input_path),
            "--output-file",
            str(output_path),
        ]
        queue.put(command)

    waiting = Queue()
    with open(args.hosts) as f:
        for line in f:
            host = line.strip()
            proc = subprocess.Popen(
                ["ssh", "-i", "~/.ssh/id_ed_25519", "-o", "StrictHostKeyChecking=no", host, "true"],
                stderr=subprocess.PIPE,
            )
            if proc.wait() != 0:
                logger.error(f"Host {host} is not available")
                continue
            logger.info(f"Host {host} is available")
            waiting.put(line.strip())

    logger.info(f"Available hosts: {waiting.qsize()}")

    running = []
    while not queue.empty() or len(running) > 0:
        while not waiting.empty() and not queue.empty():
            host = waiting.get()
            command = queue.get()
            logger.info(f"Running {command} on {host}")
            proc = subprocess.Popen(
                ["ssh", "-i", "~/.ssh/id_ed_25519", "-o", "StrictHostKeyChecking=no", host] + command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            running.append((host, proc))

        for host, proc in running:
            if proc.poll() is not None:
                if proc.returncode != 0:
                    error_message = proc.stderr.read().decode("utf-8")
                    logger.error(f"Failed {proc.args} on {host}: {error_message}")
                    queue.put(proc.args[6:])
                else:
                    logger.info(f"Finished {proc.args} on {host}")
                running.remove((host, proc))
                waiting.put(host)
                break

        sleep(1)

    logger.info("All done")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    main()
