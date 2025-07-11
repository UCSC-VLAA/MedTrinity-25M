#!/bin/bash
# checkpoint=$1
# answer_parent_path=$2

python llava/eval/run_med_caption_batch.py \
    --model-path /data3/yxie/data/checkpoints/checkpoint-3500 \
    --image-folder /data3/yxie/mimic_cxr_test_2/ \
    --question-file /data3/yxie/mimic_cxr_test_2/metadata.jsonl \
    --answers-file /data3/yxie/data/output/mimic_test.jsonl \
    --temperature 0.5 \
    --num-chunks 8 \
    --max_new_tokens 1024 \
    --batch_size 1 \
    --num_workers 8
