#!/bin/bash

work_dir=/home/shared/experiments/0118_dedup_corpusv4_ja
env_dir=environment
venv_dir=.venv
script_dir=${work_dir}/scripts

# pyenv: python version 3.10.14

cd $work_dir || exit

mkdir -p $env_dir
cd $env_dir || exit
python3.10 -m venv $venv_dir

source $venv_dir/bin/activate
pip install --upgrade --no-cache-dir pip uv
uv init
uv add --no-cache -r ${script_dir}/installer/requirements.txt 
