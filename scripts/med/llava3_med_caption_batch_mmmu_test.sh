#!/bin/bash
# checkpoint=$1
# answer_parent_path=$2

python llava/eval/run_med_caption_batch.py \
    --model-path /data3/yxie/MedTrinity-25M/checkpoints/llava-llama-med-8b-stage2-finetune-slake_orift \
    --image-folder /data3/yxie/MMMU/health_test \
    --question-file /data3/yxie/MMMU/health_test/metadata.jsonl \
    --answers-file /data3/yxie/data/output/MMMU_test_10.jsonl \
    --temperature 1.0 \
    --num-chunks 8 \
    --max_new_tokens 1024 \
    --batch_size 4 \
    --num_workers 8
