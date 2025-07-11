#!/bin/bash
# checkpoint=$1
# answer_parent_path=$2

python llama/eval/run_caption_reformat_batch.py \
    --model-path ../Llama-3-8B-Instruct \
    --question-file ../Reformat_VQA/Captions/25M_merge_shard/part_1/metadata.jsonl \
    --answers-file ../Reformat_VQA/VQAs/25M_merge_shard_part_1_vqa.jsonl \
    --temperature 0.2 \
    --num-chunks 4 \
    --max_new_tokens 8196 \
    --batch_size 32 \
    --num_workers 4
