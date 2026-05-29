#!/bin/bash

export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1
export NCCL_IB_DISABLE=1 

torchrun --nproc_per_node=1 inference.py \
    --model_path './llama-finetuned' \
    --data_path './test.json' \
    --output_path './medqa_output_2.txt' \
    # --output_path './medbullet_output_base.txt' \
    # --model_path 'meta-llama/Llama-3.1-8B-Instruct' \