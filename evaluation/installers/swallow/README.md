[日本語版](README.ja.md)
# llm-jp-eval v1.4.1 Installation and Execution Script

Script for evaluation using llm-jp-eval v1.4.1 <br>
This includes scripts for environment setup and evaluation execution.

Note: In this version, Docker is not supported, so the Code Generation task (mbpp) will be skipped.

## Usage

### Build

The installation process requires CPU resources only.

1. Clone the repository:
```shell
git clone https://github.com/llm-jp/scripts
cd scripts/evaluation/installers/swallow
```
2. Installation:
A directory for environment setup (`~/myspace/environment`) will be created under the specified directory (`~/myspace`).
Depending on your network speed, it may take at least 20 minutes.
```shell
# For a cluster with SLURM
sbatch --partition {FIX_ME} install.sh ~/myspace
# For a cluster without SLURM
bash install.sh ~/myspace > logs/install.out 2> logs/install.err
```
3.	(Optional) Configure wandb and Hugging Face:
```shell
cd ~/myspace
source environment/venv-postprocessing/bin/activate
wandb login
huggingface-cli login
```

### Contents of the Installed Directory (~/myspace)

After installation, the following directory structure will be created:
```
~/myspace/
    run-eval.sh                 Script for running the evaluation
    scripts/                    Subscripts needed for the evaluation
    logs/                       Directory for SLURM logs
    environment/
        install.sh              Installation script used
        python/                 Python runtime environment
        scripts/                Various setup scripts
        src/                    Libraries downloaded individually
        venv/                   Python virtual environment (linked to python/)
```

### Evaluation

Modify the variables in the `run-eval.sh` files as needed:

VRAM needs to be 2.5-3.5 times the model size (e.g., 13B model -> 33GB-45GB).
Memory usage will be automatically estimated from model size.
When running in a SLURM environment, the default is `--gpus 1`, so adjust to the appropriate size for your cluster along with `--mem`.
```shell
cd ~/myspace
# (Optional) If you need to change variables
cp run-eval.sh run-eval_custom.sh
# Set `resources/config_custom.yaml` in run_llm-jp-eval_custom.sh

# For a cluster with SLURM
sbatch --partition {FIX_ME} run_llm-jp-eval.sh {path/to/model} {wandb.run_name} {model size in billion}
# For a cluster without SLURM
CUDA_VISIBLE_DEVICES={FIX_ME} bash run_llm-jp-eval.sh {path/to/model} {wandb.run_name} {model size in billion}
```

#### Sample Code
```shell
# Evaluate 70B model on a cluster with SLURM using H100 (VRAM: 80GB)
sbatch --partition {FIX_ME} --gpus 4 --mem 8G run_llm-jp-eval.sh sbintuitions/sarashina2-70b test-$(whoami) 70
# Evaluate 13B model on a cluster without SLURM using A100 (VRAM: 40GB)
CUDA_VISIBLE_DEVICES=0,1 bash run_llm-jp-eval.sh llm-jp/llm-jp-13b-v2.0 test-$(whoami) 13
```

