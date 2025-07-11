import json


# output_file = "/data5/yunfei/Data/answer_file/ds_50k/ds50k_answer_file_2024_05_20_08_53_23.jsonl"
# with open(output_file, 'w') as outfile:
#     for idx in range(4):
#         with open(output_file.split(".")[0]+f"-chunk{idx}.jsonl") as infile:
#             outfile.write(infile.read())
            
# print("done!")

count = 0
with open("/data5/yunfei/Data/answer_file/ds_50k/ds50k_answer_file_2024_05_20_08_53_23.jsonl", "r") as f:
    # for line in f:
    #     # 解析每一行为JSON对象
    #     try:
    #         data = json.loads(line)
    #         # 处理每个JSON对象
    #         print(data)
    #     except json.JSONDecodeError as e:
    #         # 如果解析错误，打印错误信息
    #         print(f"Error parsing JSON line: {e}")
    for line in f:
        # 如果行不是空的，则计数增加
        if line.strip():
            count += 1
print(count)
with open("/data5/yunfei/Data/ds_50k/finetune_50k_new_8_rag_test_fix_delete.json", "r") as f:
    print(len(json.load(f)))