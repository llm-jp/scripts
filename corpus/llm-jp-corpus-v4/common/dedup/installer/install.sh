#!/bin/bash

work_dir=/model/ytsuta/workspace-model/space-dedup_analysis
env_dir=${work_dir}/environment
venv_dir=${env_dir}/.venv
src_dir=${env_dir}/src
script_root=${work_dir}/scripts/corpus/llm-jp-corpus-v4/common/dedup

export UV_PROJECT_ENVIRONMENT=$venv_dir

# create environment
cd $work_dir || exit
mkdir -p $env_dir
cd $env_dir || exit

python3 -m venv $venv_dir
source $venv_dir/bin/activate
pip install --upgrade --no-cache-dir pip uv
uv init

# install requirement
uv add --no-cache -r ${script_root}/installer/requirements.txt

# install customized datatrove
mkdir -p $src_dir
cd $src_dir || exit
git clone https://github.com/huggingface/datatrove.git -b v0.4.0
cd datatrove || exit
patch -p1 <${script_root}/installer/datatrove_diff.patch
uv pip install --no-cache-dir ".[io,processing,cli]"
