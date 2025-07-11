#!/bin/bash
# checkpoint=$1
# answer_parent_path=$2

python llava/eval/run_med_caption_batch.py \
    --model-path model_path \
    --image-folder imgs \
    --question-file question.jsonl \
    --answers-file caption.jsonl \
    --temperature 0.1 \
    --num-chunks 4 \
    --max_new_tokens 1024 \
    --batch_size 13 \
    --num_workers 4
