# FP8 check scripts

This directory contains several scripts to check the behavior of FP8 operations on Megatron-LM with existing checkpoints.

## Prerequisites

The following directories and contents must be exist before running scripts in this directory:

* `checkpoints/3.8b/base_{iter:07d}`: Megatron-LM checkpoints of the base models
* `environment`: Training environment created by the installer
* `eval_environment`: Evaluation environment created by the installer
* `outputs`: Slurm log directory
* `scripts`: Clone of this repository

## Scripts

All scripts must be invoked from the root of the experiment directory.

* `run_train.sh`: Runs cont'd training with several configurations
* `run_convert.sh`: Runs model conversion to Hugging Face format
* `run_eval.sh`: Runs llm-jp-eval 1.3.1 evaluation
