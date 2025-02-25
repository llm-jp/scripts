#!/bin/bash

work_dir=/model/experiments/0118_dedup_corpusv4_ja
env_dir=${work_dir}/environment
src_dir=${env_dir}/src
venv_dir=${env_dir}/.venv
script_root=${work_dir}/scripts/corpus/llm-jp-corpus-v4/common/dedup

# pyenv: python version >= 3.10

# create environment
cd $work_dir || exit
mkdir -p $env_dir
cd $env_dir || exit

python3.10 -m venv $venv_dir
source $venv_dir/bin/activate
pip install --upgrade --no-cache-dir pip uv
uv init

# install customized datatrove
mkdir -p $src_dir
cd $src_dir || exit
git clone https://github.com/huggingface/datatrove.git -b v0.4.0
cd datatrove || exit
patch -p1 < ${script_root}/installer/datatrove_diff.patch
uv pip install --no-cache-dir  ".[io,processing,cli]"

# install others
uv add --no-cache -r ${script_root}/installer/requirements.txt 
