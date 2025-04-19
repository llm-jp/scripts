#!/bin/bash

. ./scripts/scripts/estimate_vllm_memory_usage.sh

gpu_proportion=""
tp_size=""
dp_size=""
estimate_vllm_memory_tpdp $1 gpu_proportion tp_size dp_size

echo $gpu_proportion $tp_size $dp_size
