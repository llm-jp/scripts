# Megatron-LM Installer for LLM-jp on B200 (Blackwell / sm_100)

Native (no-container) `uv` environment for **upstream Megatron-LM** on the B200
Slurm cluster. Port of `v4-upstream-megatron-abci`, adapted to Slurm + CUDA 13 +
Blackwell.

## Stack

| Component | Version |
|---|---|
| Python | 3.12 (uv-managed) |
| CUDA toolkit | 13.0.3 (system, for nvcc/APEX) |
| torch | 2.12.0 + cu130 |
| Transformer Engine | 2.16.0 (prebuilt cu13 wheel) |
| FlashAttention | **4** (`flash-attn-4` 4.0.0b17 + cutlass-dsl 4.5.2 + quack 0.5.0) |
| APEX | `becbb77cea4cb54f2929f7c938a0a6f7dd1fdc39`, built for sm_100 |
| Megatron-LM | `llm-jp/Megatron-LM` @ `core_v0.18.0_b200` |
| Tokenizer | `llm-jp/llm-jp-tokenizer` @ `v3.0b2` |

## Usage

Build (submits to the Slurm `cpu` partition — it has internet + cores):

```bash
cd pretrain/installers/v4-upstream-megatron-b200/
bash run_setup.sh <env_install_path>
```

Or run interactively on a cpu node:

```bash
srun -p cpu -c 64 --pty env TARGET_DIR=<env_install_path> bash sbatch_setup.sh
```

Verify on a GPU node (B200):

```bash
TARGET_DIR=<env_install_path> srun -p gpu --gres=gpu:1 -c 16 bash verify/verify_env.sh
```

Activate for training:

```bash
source <env_install_path>/scripts/environment.sh   # version pins + CUDA toolkit
source <env_install_path>/venv/bin/activate         # Python venv
```

## Tunables (override before `run_setup.sh`, or edit `scripts/environment.sh`)

- `PRETRAIN_MEGATRON_REPO` / `PRETRAIN_MEGATRON_TAG` — Megatron-LM source/branch.
- `PRETRAIN_TORCH_VERSION`, `PRETRAIN_TRANSFORMER_ENGINE_VERSION`, etc. — pins.
- `PRETRAIN_CUDA_HOME` — system CUDA toolkit used for nvcc/APEX.
- `PRETRAIN_APEX_COMMIT` — override the pinned APEX commit.
