#!/bin/bash

work_dir=/model/experiments/0118_dedup_corpusv4_ja
env_dir=environment
venv_dir=.venv
script_root=${work_dir}/scripts/corpus/llm-jp-corpus-v4/common/dedup

# pyenv: python version 3.10.14

cd $work_dir || exit

mkdir -p $env_dir
cd $env_dir || exit
python3.10 -m venv $venv_dir

source $venv_dir/bin/activate
pip install --upgrade --no-cache-dir pip uv
uv init
uv add --no-cache -r ${script_root}/installer/requirements.txt 
