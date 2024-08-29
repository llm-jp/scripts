# LLM Evaluation using g-leaderboard (GENIAC Official Evaluation)

This repository contains scripts for evaluating LLMs using [g-leaderboard](https://github.com/wandb/llm-leaderboard/tree/g-leaderboard).

## Usage

### Build

Clone this repository and move to the installation directory.

```bash
git clone https://github.com/llm-jp/scripts
cd scripts/evaluation/installers/g-leaderboard
```

Then, run the installation script.
The following command will create an installation directory under the specified directory (here, `~/g-leaderboard`).
`<env-name>` should be the name of the environment (llm-jp, llm-jp-nvlink, sakura, etc).
The list of available environment names can be found in the `scripts/envs` directory.

```bash
# For a cluster with SLURM
sbatch --partition {partition} install.sh {env-name} ~/g-leaderboard
# For a cluster without SLURM
bash install.sh {env-name} ~/g-leaderboard > logs/install.out 2> logs/install.err
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
    run_g-leaderboard.sh      Script for running g-leaderboard
    logs/                     Log files written by SLURM jobs
    resources/
        config_base.yaml      Configuration file template
    environment/
        installer_envvar.log  List of environment variables recorded during installation
        install.sh            Installation script
        python/               Python built from source
        scripts/              Scripts for environment settings
        src/                  Downloaded libraries
        venv/                 Python virtual environemnt (linked to python/)
```

### Evaluation

The evaluation script takes the model path and wandb run name as arguments.
For the other settings, edit the configuration file `resources/config_base.yaml` and/or `resources/config_custom.yaml`.
 - To edit the tokenizer, wandb entity, and/or wandb project: Edit `run_g-leaderboard.sh`.
 - Otherwise: Edit `resources/config_base.yaml` and `run_g-leaderboard.sh`.

```shell
cd ~/g-leaderboard
# For a cluster with SLURM
AZURE_OPENAI_ENDPOINT=xxx AZURE_OPENAI_KEY=xxx sbatch --partition {partition} run_g-leaderboard.sh {path/to/model} {wandb.run_name}
# For a cluster without SLURM
CUDA_VISIBLE_DEVICES=<num> AZURE_OPENAI_ENDPOINT=xxx AZURE_OPENAI_KEY=xxx bash run_g-leaderboard.sh {path/to/model} {wandb.run_name}
```

#### Sample code

```shell
# For a cluster with SLURM
AZURE_OPENAI_ENDPOINT=xxx AZURE_OPENAI_KEY=xxx sbatch --partition {partition} run_g-leaderboard.sh llm-jp/llm-jp-13b-v2.0 g-leaderboard-$(whoami)
# For a cluster without SLURM
AZURE_OPENAI_ENDPOINT=xxx AZURE_OPENAI_KEY=xxx bash run_g-leaderboard.sh llm-jp/llm-jp-13b-v2.0 g-leaderboard-$(whoami)
```

### About Azure OpenAI API

To conduct an evaluation, you must configure the Azure OpenAI API by setting the endpoint and key for the deployment named `gpt-4`, which corresponds to `gpt-4-0613`. Please contact the administrator to obtain the necessary endpoint and key.
