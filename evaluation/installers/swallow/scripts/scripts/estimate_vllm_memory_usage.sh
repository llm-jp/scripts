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
