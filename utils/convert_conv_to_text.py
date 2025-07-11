from llava.conversation import conv_templates
import json
import os


def QA2Text(example):
    cap = example["conversations"][0]["value"]
    ans = example["conversations"][1]["value"]
    conv = conv_templates["llama3_qa"].copy()
    conv.append_message(conv.roles[0], cap)
    conv.append_message(conv.roles[1], ans)
    prompt = conv.get_prompt()
    example["text"] = prompt.replace("<image>\n", "")
    example.pop("conversations")
    example.pop("id")
    example.pop("image")
    example.pop("source")
    
    return example


with open("/data5/yunfei/Reformat_VQA/VQAs/selected_samples_finetuning_newprompt.jsonl", "r") as f:
    metadata = [json.loads(line) for line in f]
    
with open("/data5/yunfei/Reformat_VQA/VQAs/llama3_finetune_text.jsonl", "w") as f:
    for data in metadata:
        f.write(json.dumps(QA2Text(data)))
        f.write("\n")