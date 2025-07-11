export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

checkpoint=$1
answer_parent_path=$2

current_datetime=$(date +"%Y_%m_%d_%H_%M_%S")


# python llava/eval/run_med_datasets_eval_batch.py --num-chunks  6 --model-name $checkpoint \
#     --question-file ../Data/medical_data/VQA-RAD/test.json \
#     --image-folder ../Data/medical_data/VQA-RAD/images \
#     --answers-file "$answer_parent_path/VQA-RAD/vqa_rad_test_answer_file_$current_datetime.jsonl" && \

# python llava/eval/run_eval_nocandi.py \
#     --gt ../Data/medical_data/VQA-RAD/test.json \
#     --pred "$answer_parent_path/VQA-RAD/vqa_rad_test_answer_file_$current_datetime.jsonl"

# python llava/eval/run_med_datasets_eval_batch.py --num-chunks 6  --model-name $checkpoint \
#     --question-file ../Data/medical_data/SLAKE/test.json \
#     --image-folder ../Data/medical_data/SLAKE/imgs \
#     --answers-file "$answer_parent_path/SLAKE/slake_test_answer_file_$current_datetime.jsonl" && \

# python llava/eval/run_eval_nocandi.py \
#     --gt ../Data/medical_data/SLAKE/test.json \
#     --pred "$answer_parent_path/SLAKE/slake_test_answer_file_$current_datetime.jsonl"

# python llava/eval/run_med_datasets_eval_batch.py --num-chunks 8  --model-name $checkpoint \
#     --question-file ../Data/medical_data/Path-VQA/test.json \
#     --image-folder ../Data/medical_data/Path-VQA/images \
#     --answers-file "$answer_parent_path/Path-VQA/pathvqa_answer_file_$current_datetime.jsonl" && \

# python llava/eval/run_eval_nocandi.py \
#     --gt ../Data/medical_data/Path-VQA/test.json \
#     --pred "$answer_parent_path/Path-VQA/pathvqa_answer_file_$current_datetime.jsonl"

python llava/eval/run_med_datasets_eval_batch.py --num-chunks 4  --model-name $checkpoint \
    --question-file ../Data/ds_50k/finetune_50k_new_8_rag_test_fix_delete.json \
    --image-folder ../Data/ds_50k/w_mask \
    --answers-file "$answer_parent_path/ds_50k/ds50k_answer_file_$current_datetime.jsonl" && \

python llava/eval/run_eval_nocandi.py \
    --gt ../Data/ds_50k/finetune_50k_new_8_rag_test_fix_delete.json \
    --pred "$answer_parent_path/ds_50k/ds50k_answer_file_$current_datetime.jsonl"