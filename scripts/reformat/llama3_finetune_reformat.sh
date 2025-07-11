#!/bin/bash
# export PYTHONPATH="${PYTHONPATH}:/usr/local/anaconda3/envs/llava/bin/python" 
export TOKENIZERS_PARALLELISM=false 
# export NCCL_P2P_DISABLE=1
torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 llama/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --model_path /data5/yunfei/Llama-3-8B-Instruct \
    --data_file ../Reformat_VQA/VQAs/llama3_finetune_text.jsonl \
    --gradient_checkpointing True \
    --bf16 True \
    --new_model Llama-3-8B-Instruct-reformat \
    --output_dir ./llama3/Llama-3-8B-Instruct-reformat \
    --optim "paged_adamw_32bit" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --max_grad_norm 0.3 \
    --group_by_length True \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --max_seq_length 4096 \
    --gradient_checkpointing True \
    --report_to wandb
