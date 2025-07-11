import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import transformers
from dataclasses import dataclass, field

from typing import List, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    logging,
)

from transformers.generation.stopping_criteria import StopStringCriteria, EosTokenCriteria, StoppingCriteriaList

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, StopTokenCriteria
from torch.utils.data import Dataset, DataLoader

# from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, captions, tokenizer):
        self.captions = captions
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        line = self.captions[index]
        qs = line["caption"]

        conv = conv_templates["llama3_qa"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt().replace("<image>\n", "")

        # input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        return index, prompt

    def __len__(self):
        return len(self.captions)


# @dataclass
# class DataCollatorForTextGeneration(object):
#     tokenizer: transformers.PreTrainedTokenizer

#     def pad_sequence(self, input_ids, batch_first, padding_value):
#         if self.tokenizer.padding_side == "left":
#             input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids] 
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids,
#             batch_first=batch_first,
#             padding_value=padding_value)
#         if self.tokenizer.padding_side == "left":
#             input_ids = torch.flip(input_ids, [1])
#         return input_ids

#     def __call__(self,
#                  batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
#         indices, input_ids= zip(*batch)
#         input_ids = self.pad_sequence(
#             input_ids,
#             batch_first=True,
#             padding_value=self.tokenizer.eos_token_id)

#         return indices, input_ids

# DataLoader
def create_data_loader(questions, tokenizer, batch_size=1, num_workers=4):
    dataset = CustomDataset(questions, tokenizer)
    # collator = DataCollatorForTextGeneration(tokenizer=tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    # disable_torch_init()
    # model_path = os.path.expanduser(args.model_path)
    # model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, use_flash_attn=True)
    
    model_path = args.model_path
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", attn_implementation="flash_attention_2", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # set padding side to `left` for batch text generation
    tokenizer.padding_side = "left"

    if args.question_file.endswith('.jsonl'):
        with open(args.question_file, 'r') as f:
            questions = [json.loads(line) for line in f]          
    elif args.question_file.endswith('.json'):
        questions = [q for q in json.load(open(os.path.expanduser(args.question_file), "r"))]
    answers_file = os.path.expanduser(args.answers_file)

    if os.path.exists(answers_file):
        origin_q_num = len(questions)
        experiment_name_with_split = args.answers_file.split('-chunk')[0]
        answered_ids = set()
        for idx in range(args.num_chunks):
            if os.path.exists(f"{experiment_name_with_split}-chunk{idx}.jsonl"):
                with open(f"{experiment_name_with_split}-chunk{idx}.jsonl") as infile:
                    answered_ids.update(json.loads(line)["question_id"] for line in infile)
        
        id_name = "id" if "id" in questions[0] else "question_id"
        
        questions = [q for q in questions if q[id_name] not in answered_ids]
        print(f"already answered question num: {len(answered_ids)}, origin question num: {origin_q_num}, now question num: {len(questions)}")
            
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "a")

    data_loader = create_data_loader(
        questions,
        tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    data_loader = iter(data_loader)

    conv = conv_templates["llama3_qa"].copy()
    stop_str = conv.sep
    
    for indices, prompts in tqdm(data_loader):
        try:
            with torch.inference_mode():
                inputs = tokenizer(prompts, return_tensors="pt", padding=True).to('cuda')
                output_ids = model.generate(
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([StopTokenCriteria(128001, 128009)]),
                    **inputs
                )
              
            # only get the generated ids  
            input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
            generated_ids = output_ids[:, input_length:]

            outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

            for index, output in zip(indices, outputs):
                line = questions[index]
                idx = line["question_id"] if 'question_id' in line else line["id"]
                image = line["file_name"]
                cur_prompt = line["caption"]
                # ans_id = shortuuid.uuid()
                ans_file.write(json.dumps({
                    "question_id": idx,
                    "image": image,
                    "caption": cur_prompt,
                    "qa": output.strip(),
                    # "answer_id": ans_id,
                }) + "\n")
            ans_file.flush()
        except Exception as e:
            print(f"Error processing batch with indices {indices}: {e}")
            continue

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    eval_model(args)