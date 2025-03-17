[日本語版](README.ja.md)
# `swallow-evaluation (en/code)` Installation and Execution Script

Script for evaluation using [swallow-evaluation](https://github.com/swallow-llm/swallow-evaluation) <br>
This includes scripts for environment setup and evaluation execution.

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

### Build with code generation
Docker is required to evaluate code generation.
```shell
# For a cluster with SLURM
sbatch --partition {FIX_ME} install-code.sh ~/myspace
# For a cluster without SLURM
bash install-code.sh ~/myspace > logs/install.out 2> logs/install.err
```

3. Configure wandb and Hugging Face:
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
        venv-harness/           Python virtual environment for lm-evaluation-harness-en (linked to python/)
        venv-bigcode/           Python virtual environment for bigcode-evaluation-harness (linked to python/)
        venv-postprocessing/    Python virtual environment for postprocessing (linked to python/)
```

### Evaluation

Modify the variables in the `run-eval.sh` files as needed:

VRAM needs to be 2.5-3.5 times the model size (e.g., 13B model -> 33GB-45GB).
Memory usage will be automatically estimated from model size.
When running in a SLURM environment, the default is `--gpus 1`, so adjust to the appropriate size for your cluster along with `--mem`.

> [!NOTE]
> Currently, only the WITHOUT CODE GENERATION setting is supported for specifying output directories.

```shell
# For a cluster with SLURM
sbatch --partition {FIX_ME} run-eval.sh {path/to/model} {wandb.run_name} {output/dir} {OPTIONAL: model size in billion}
# For a cluster without SLURM
CUDA_VISIBLE_DEVICES={FIX_ME} bash run-eval.sh {path/to/model} {wandb.run_name} {output/dir} {OPTIONAL: model size in billion}
```

#### Sample Code
```shell
# Evaluate 70B model on a cluster with SLURM using H100 (VRAM: 80GB)
sbatch --partition {FIX_ME} --gpus 4 --mem 8G run-eval.sh sbintuitions/sarashina2-70b test-$(whoami) ./test-$(whoami) 70
# Evaluate 13B model on a cluster without SLURM using A100 (VRAM: 40GB)
CUDA_VISIBLE_DEVICES=0,1 bash run-eval.sh llm-jp/llm-jp-13b-v2.0 test-$(whoami) ./test-$(whoami) 13
```

