# Runtime environment for the sakura B200 trainer (sourced by sbatch_train.sh).
# Activates the native env and puts the cu13 wheel libs on the loader path so TE
# finds libcudart/libcublas/cuDNN at dlopen (mirrors verify/smoke_qwen3 setup).

# ENV_DIR must be exported by the caller.
: "${ENV_DIR:?ENV_DIR must be set}"

# Python env + build-time CUDA toolkit pins.
source "${ENV_DIR}/scripts/environment.sh"
source "${ENV_DIR}/venv/bin/activate"

# Runtime: cu13 wheels (not the system toolkit) supply CUDA libs to TE.
PY="${ENV_DIR}/venv/bin/python"
LD_EXTRA=$("${PY}" - <<'PY'
import os, glob, torch
sp = os.path.dirname(os.path.dirname(torch.__file__))
d = glob.glob(os.path.join(sp, "nvidia", "*", "lib")); d.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
print(":".join(d))
PY
)
export LD_LIBRARY_PATH="${LD_EXTRA}:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${ENV_DIR}/venv/lib/python3.12/site-packages/nvidia"
export CUDA_PATH="${CUDA_HOME}"
export NVTE_FRAMEWORK=pytorch

# Distributed rendezvous (torchrun reads these via sbatch_train.sh).
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=$((10000 + (${SLURM_JOB_ID:-0} % 50000)))
NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
NUM_GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NODES=${NUM_NODES} GPUS=${NUM_GPUS}"

ulimit -n 65536 1048576

# Logging / NCCL. NOTE: do NOT set CUDA_DEVICE_MAX_CONNECTIONS=1 — Blackwell does
# not need it and it conflicts with the distributed optimizer's comm overlap.
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
# InfiniBand: more QPs per connection improves multi-rail RDMA throughput.
export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-8}
# Reduce allocator fragmentation (frees a few GiB of headroom on first steps).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
