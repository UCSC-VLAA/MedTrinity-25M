python utils/reformat_openai_batch_call.py \
    --caption-file ../Reformat_VQA/Captions/25M_merge_shard/part_1/metadata.jsonl \
    --reformat-file ../Reformat_VQA/Openai_batch_formats/test_25M_merge_shard_part_1_vqa.jsonl \
    --model gpt-3.5-turbo-0125 \
    --max_tokens 2048 \
