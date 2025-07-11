#!/bin/bash
checkpoint=./checkpoints/llava-llama-med-8b-stage2-finetune


python llava/eval/run_med_datasets_eval_batch.py --num-chunks  8 --model-name $checkpoint \
    --question-file ../Data/medical_data/VQA-RAD/test.json \
    --image-folder ../Data/medical_data/VQA-RAD/images \
    --answers-file ../Data/answer_fie/VQA-RAD/vqa_rad_modeltest_answer_file_$current_datetime.jsonl && \

python llava/eval/run_eval_nocandi.py \
    --gt ../Data/medical_data/VQA-RAD/test.json \
    --pred ../Data/answer_fie/VQA-RAD/vqa_rad_modeltest_answer_file_$current_datetime.jsonl 

