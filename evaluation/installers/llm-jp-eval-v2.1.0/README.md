[日本語版](README.ja.md)
# llm-jp-eval v2.1.0 Installation and Execution Script

Script for evaluation using llm-jp-eval v2.1.0 <br>
This includes scripts for environment setup and evaluation execution.

Note: In this version, Docker is not supported, so the Code Generation task (mbpp) will be skipped.

## Usage

### Build

The installation process requires CPU resources only.

1. Clone the repository:
```shell
git clone https://github.com/llm-jp/scripts
cd scripts/evaluation/installers/llm-jp-eval-v2.1.0
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
source environment/venv/bin/activate
wandb login
huggingface-cli login
```

### Contents of the Installed Directory (~/myspace)

After installation, the following directory structure will be created:
```
~/myspace/
    run_llm-jp-eval.sh         Script for running the evaluation
    logs/                      Directory for SLURM logs
    resources/
        config_base.yaml       Template configuration file for evaluation
    vllm_outputs/              Output directory for vllm
    environment/
        installer_envvar.log   Log of environment variables recorded after installation started
        install.sh             Installation script used
        dataset/llm-jp-eval    Dataset for llm-jp-eval evaluation
        python/                Python runtime environment
        scripts/               Various setup scripts
        src/                   Libraries downloaded individually
        venv/                  Python virtual environment (linked to python/)
```

### Evaluation

Modify the variables in `run_llm-jp-eval.sh` or the `resources/config_*.yaml` files as needed:
- If you want to change the tokenizer, wandb entity, or wandb project, modifying only `run_llm-jp-eval.sh` will suffice.
- For other changes, modify `resources/config_*.yaml` and specify the file in `run_llm-jp-eval.sh`.

VRAM needs to be 2.5-3.5 times the model size (e.g., 13B model -> 33GB-45GB).
If evaluating on more than 2 GPUs, you need to modify `tensor_parallel_size` in `resources/config_offline_inference_vllm.yaml`.
When running in a SLURM environment, the default is `--gpus 1`, so adjust to the appropriate size for your cluster along with `--mem`.
```shell
cd ~/myspace
# (Optional) If you need to change variables
cp resources/config_base.yaml resources/config_custom.yaml
cp run_llm-jp-eval.sh run_llm-jp-eval_custom.sh
# Set `resources/config_custom.yaml` in run_llm-jp-eval_custom.sh

# For a cluster with SLURM
sbatch --partition {FIX_ME} run_llm-jp-eval.sh {path/to/model} {wandb.run_name}
# For a cluster without SLURM
CUDA_VISIBLE_DEVICES={FIX_ME} bash run_llm-jp-eval.sh {path/to/model} {wandb.run_name}
```

#### Sample Code
```shell
# Evaluate 70B model on a cluster with SLURM using H100 (VRAM: 80GB)
sbatch --partition {FIX_ME} --gpus 4 --mem 8G run_llm-jp-eval.sh sbintuitions/sarashina2-70b test-$(whoami)
# Evaluate 13B model on a cluster without SLURM using A100 (VRAM: 40GB)
CUDA_VISIBLE_DEVICES=0,1 bash run_llm-jp-eval.sh llm-jp/llm-jp-13b-v2.0 test-$(whoami)
```

## For Developers: Command to Create resources/sha256sums.csv
```shell
TARGET_DIR={path/to/dataset/directory/containing/json/files}
find $TARGET_DIR -type f | xargs -I{} sh -c 'echo -e "$(basename {})\t$(sha256sum {} | awk "{print \$1}")"'
```
