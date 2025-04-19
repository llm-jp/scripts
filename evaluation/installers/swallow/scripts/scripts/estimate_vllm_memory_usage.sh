#!/bin/bash
estimate_vllm_memory() {
    local model_params_billion=$1  # Number of model parameters in billion
    local precision=${2:-"fp16"}   # Default to FP16, can be "fp32"

    # Memory required per parameter (in bytes)
    local bytes_per_param=2  # FP16 (half-precision)
    if [[ "$precision" == "fp32" ]]; then
        bytes_per_param=4  # FP32 (full-precision)
    fi

    # Total model memory in GB
    local model_memory_gb=$(( model_params_billion * bytes_per_param  ))

    # Overhead (~20% extra for activations, KV cache, optimizations)
    local overhead_gb=$(( model_memory_gb / 3 ))
    local total_required_gb=$(( model_memory_gb + overhead_gb ))

    # Get total GPU memory in GB
    local total_gpu_memory_gb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum / 1024}')
    # echo $total_gpu_memory_gb
    # echo "sep"

    # Estimate proportion needed, set minimum propotion as 0.2 to prevent insufficient memory for KV cache
    local proportion_needed=$(echo "scale=2; $total_required_gb / $total_gpu_memory_gb" | bc)
    proportion_needed=$(echo "if ($proportion_needed < 0.2) 0.2 else $proportion_needed" | bc)

    echo "$proportion_needed"
}

estimate_vllm_memory_tpdp() {
    local -n out_gpu_proportion=$2
    local -n out_tp_size=$3
    local -n out_dp_size=$4

    local model_params_billion=$1  # Number of model parameters in billion
    local precision=${2:-"fp16"}   # Default to FP16, can be "fp32"

    # Memory required per parameter (in bytes)
    local bytes_per_param=2  # FP16 (half-precision)
    if [[ "$precision" == "fp32" ]]; then
        bytes_per_param=4  # FP32 (full-precision)
    fi

    # Total model memory in GB
    local model_memory_gb=$(( model_params_billion * bytes_per_param  ))

    # Overhead (~20% extra for activations, KV cache, optimizations)
    local overhead_gb=$(( model_memory_gb / 5 ))
    local total_model_gb=$(( model_memory_gb + overhead_gb ))

    # Get total GPU memory in GB
    local total_gpu_memory_gb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum / 1024}')
    local single_gpu_memory_gb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | awk '{print $1 / 1024}')
    local total_num_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    local targeted_per_gpu_memory_gb=$(awk "BEGIN {print $single_gpu_memory_gb / 1.8}") #  $(( single_gpu_memory_gb / 2 ))
    echo "single gpu memory" $single_gpu_memory_gb
    local targeted_tp_size=$(awk "BEGIN {res = $total_model_gb * 1.25 / $targeted_per_gpu_memory_gb; ceil = (res == int(res)) ? int(res) : int(res)+1; print ceil} ")
    echo "tp size" $targeted_tp_size
    # local possible_tp_size=$(( total_model_gb * 1.33 / targeted_per_gpu_memory_gb ))
    local possible_dp_size=$(( total_num_gpu / targeted_tp_size ))
    local proportion_needed=$(awk "BEGIN {print $total_model_gb * 1.25 / $targeted_tp_size / $single_gpu_memory_gb}")
    proportion_needed=$(echo "if ($proportion_needed < 0.3) 0.3 else $proportion_needed" | bc)

    if [[ $possible_dp_size -eq 0 ]]; then
        echo "INSUFFICIENT GPU MEMORY. DP SIZE is ZERO";
        exit 1
    fi

    if [[ $targeted_tp_size -gt 1 ]]; then
        possible_dp_size=1
    fi

    
    out_tp_size=$targeted_tp_size
    out_dp_size=$possible_dp_size
    out_gpu_proportion=$proportion_needed

    # echo "$proportion_needed"
}
