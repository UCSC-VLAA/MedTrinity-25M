#!/bin/bash

model_name_or_path=/data3/yxie/LLaVA-Med/checkpoints/llava_med_in_text_60k_ckpt2
checkpoint=./checkpoints/llava_med_ori_mimic_noourc


torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $model_name_or_path \
    --version llama2 \
    --data_path /data3/yxie/mimic_cxr_test_ind/metadata.jsonl \
    --image_folder /data3/yxie/mimic_cxr_test_ind \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --gradient_checkpointing True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $checkpoint \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb 

# python llava/eval/run_med_datasets_eval_batch.py --num-chunks 4 --model-name $checkpoint \
#     --question-file ../Data/medical_data/VQA-RAD/test.json \
#     --image-folder ../Data/medical_data/VQA-RAD/images \
#     --answers-file ../Data/answer_fie/VQA-RAD/vqa_rad_modeltest_answer_file_$current_datetime.jsonl && \

# python llava/eval/run_eval_nocandi.py \
#     --gt ../Data/medical_data/VQA-RAD/test.json \
#     --pred ../Data/answer_fie/VQA-RAD/vqa_rad_modeltest_answer_file_$current_datetime.jsonl