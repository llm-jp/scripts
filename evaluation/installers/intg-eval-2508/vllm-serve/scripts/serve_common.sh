#!/bin/bash
# Helpers to manage a local `vllm serve` process shared by multiple
# evaluation clients (swallow harness / llm-jp-eval). Sourced, not executed.
#
# Contract:
#   start_vllm_server VENV_DIR MODEL PORT TP GPU_MEM_UTIL LOG_FILE [EXTRA_ARGS...]
#     - launches `vllm serve` in the background; sets VLLM_SERVER_PID
#   wait_vllm_server PORT TIMEOUT_SEC
#     - blocks until GET /health returns 200; dies on timeout or server death
#   stop_vllm_server
#     - terminates the server process group (idempotent)

start_vllm_server() {
    local venv_dir=$1 model=$2 port=$3 tp=$4 gpu_mem_util=$5 log_file=$6
    shift 6

    # `setsid` gives the server its own process group so that stop_vllm_server
    # can also reap the worker processes vllm spawns for tensor parallelism.
    setsid "${venv_dir}/bin/vllm" serve "$model" \
        --port "$port" \
        --tensor-parallel-size "$tp" \
        --gpu-memory-utilization "$gpu_mem_util" \
        --max-logprobs 20 \
        "$@" \
        > "$log_file" 2>&1 &
    VLLM_SERVER_PID=$!
    >&2 echo "vllm server starting (pid=${VLLM_SERVER_PID}, port=${port}, log=${log_file})"
}

wait_vllm_server() {
    local port=$1 timeout_sec=${2:-1800}
    local start_ts=$(date +%s)
    until curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; do
        if ! kill -0 "$VLLM_SERVER_PID" 2>/dev/null; then
            >&2 echo "ERROR: vllm server (pid=${VLLM_SERVER_PID}) died during startup."
            return 1
        fi
        if [ $(( $(date +%s) - start_ts )) -ge "$timeout_sec" ]; then
            >&2 echo "ERROR: vllm server did not become healthy within ${timeout_sec}s."
            return 1
        fi
        sleep 5
    done
    >&2 echo "vllm server is healthy (port=${port})"
}

stop_vllm_server() {
    if [ -n "${VLLM_SERVER_PID:-}" ] && kill -0 "$VLLM_SERVER_PID" 2>/dev/null; then
        >&2 echo "stopping vllm server (pid=${VLLM_SERVER_PID})"
        # Negative PID = whole process group (tp workers included).
        kill -- "-${VLLM_SERVER_PID}" 2>/dev/null || kill "$VLLM_SERVER_PID" 2>/dev/null || true
        # Give it a moment to release GPU memory before any follow-up job.
        local i
        for i in $(seq 1 30); do
            kill -0 "$VLLM_SERVER_PID" 2>/dev/null || break
            sleep 2
        done
        kill -9 -- "-${VLLM_SERVER_PID}" 2>/dev/null || true
    fi
    VLLM_SERVER_PID=
}

find_open_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}
