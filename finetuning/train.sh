#!/bin/bash

export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1
export NCCL_IB_DISABLE=1 
export TRANSFORMERS_VERBOSITY=info

torchrun --nproc_per_node=4 train.py \
    --model_name_or_path 'meta-llama/Llama-3.1-8B-Instruct' \
    --model_max_length 2048 \
    --data_dir './data_dir' \
    --output_dir ./final_result \
    --resume_from_checkpoint True \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters False \
    --is_lora True \
    --lora_rank 128 \
    --lora_alpha 64 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --optim "adamw_torch_fused" \
    --bf16 True \
    --tf32 True \
    --num_train_epochs 4 \
    --learning_rate 1.5e-4 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_steps 1 \
    --save_steps 1000 \
    --eval_strategy "steps" \
    --eval_steps 250 \
    --save_total_limit 3 \
    --report_to "none"
