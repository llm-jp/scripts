# g-leaderboard (GENIAC Official Evaluation) installation and execution script

This repository contains scripts for evaluating LLMs using g-leaderboard.

## Usage

### Build

Clone this repository and move to the installation directory.

```bash
git clone https://github.com/llm-jp/scripts
cd scripts/evaluation/installers/g-leaderboard
```

Then, run the installation script.
The following command will create a working directory under the specified directory (`~/g-leaderboard`).
`<env-name>` should be the name of the environment (llm-jp, llm-jp-nvlink, sakura, etc).
The list of available environment names can be found in the `scripts/envs` directory.

```bash
# For a cluster with SLURM
sbatch --partition {partition} install.sh <env-name> ~/g-leaderboard
# For a cluster without SLURM
bash install.sh <env-name> ~/g-leaderboard > logs/install.out 2> logs/install.err
```

After the installation is complete, set up the wandb and huggingface accounts.

```shell
cd ~/g-leaderboard
source environment/venv/bin/activate
wandb login
huggingface-cli login
```

### Contents in installed directory (~/g-leaderboard)

The following directory structure will be created after installation.

```
~/g-leaderboard/
    run_g-leaderboard.sh  Script for running g-leaderboard
    logs/                 Log files for SLURM jobs
    resources/
        config_base.yaml  Configuration file template
    environment/
        installer_envvar.log  List of environment variables recorded during installation
        install.sh            Installation script
        python/               Python
        scripts/              Scripts for environment settings
        src/                  Downloaded libraries
        venv/                 Python virtual environemnt (linked to python/)
```

### Evaluation

Replace variables as needed in `run_g-leaderboard.sh` and `resources/config_base.yaml`.
 - To edit tokenizer, wandb entity, and/or wandb project: Edit `run_g-leaderboard.sh`.
 - To edit others: Edit `resources/config_base.yaml` and `run_g-leaderboard.sh`.

```shell
cd ~/g-leaderboard
# (Optional) If you need to change variables
cp resources/config_base.yaml resources/config_custom.yaml
cp run_g-leaderboard.sh run_g-leaderboard_custom.sh
# Set `resources/config_custom.yaml` in run_g-leaderboard_custom.sh

# For a cluster with SLURM
sbatch --partition {partition} run_g-leaderboard.sh {path/to/model} {wandb.run_name}
# For a cluster without SLURM
CUDA_VISIBLE_DEVICES={num} bash run_g-leaderboard.sh {path/to/model} {wandb.run_name}
```

#### Sample code

```shell
# For a cluster with SLURM
sbatch --partition {partition} run_g-leaderboard.sh llm-jp/llm-jp-13b-v2.0 g-leaderboard-$(whoami)
# For a cluster without SLURM
CUDA_VISIBLE_DEVICES=0 bash run_g-leaderboard.sh llm-jp/llm-jp-13b-v2.0 g-leaderboard-$(whoami)
```
