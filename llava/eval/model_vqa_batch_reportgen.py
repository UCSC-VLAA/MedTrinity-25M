import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import transformers
from dataclasses import dataclass, field

from typing import List, Tuple

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import itertools

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        qs = qs.replace('<image>', '').strip()
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return index, input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


@dataclass
class DataCollatorForVisualTextGeneration(object):
    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids] 
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=batch_first,
            padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self,
                 batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        indices, input_ids, images, image_sizes = zip(*batch)
        input_ids = self.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        images = torch.stack(images, dim=0)
        return indices, input_ids, images, [image_size for image_size in image_sizes]

# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    collator = DataCollatorForVisualTextGeneration(tokenizer=tokenizer)
    data_loader = DataLoader(dataset, collate_fn=collator, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader

def safe_json_loads(line):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        print(f"Warning: Skipping invalid JSON line: {line.strip()}")
        return None
def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, use_flash_attn=True)

    # set padding side to `left` for batch text generation
    model.config.tokenizer_padding_side = tokenizer.padding_side = "left"

    if args.question_file.endswith('.jsonl'):
        with open(args.question_file, 'r') as f:
            questions = []
            for line in f:
                try:
                    questions.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line.strip()}")
                    continue                
            
    elif args.question_file.endswith('.json'):
        questions = [q for q in json.load(open(os.path.expanduser(args.question_file), "r"))]
    answers_file = os.path.expanduser(args.answers_file)

    if os.path.exists(answers_file):
        origin_q_num = len(questions)
        experiment_name_with_split = args.answers_file.split('-chunk')[0]
        answered_ids = set()
        for idx in range(args.num_chunks):
            with open(f"{experiment_name_with_split}-chunk{idx}.jsonl") as infile:
                for line in infile:
                    parsed_line = safe_json_loads(line)
                    if parsed_line is not None and "question_id" in parsed_line:
                        answered_ids.add(parsed_line["question_id"])
        
        id_name = "id" if "id" in questions[0] else "question_id"
        
        questions = [q for q in questions if q[id_name] not in answered_ids]
        print(f"already answered question num: {len(answered_ids)}, origin question num: {origin_q_num}, now question num: {len(questions)}")
            
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "a")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(
        questions,
        args.image_folder,
        tokenizer,
        image_processor,
        model.config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_loader_iter = iter(data_loader)
    data_loader_length = len(data_loader)    
    with tqdm(total=data_loader_length) as pbar:    
        while True:
            try:
                # 获取下一个 batch
                indices, input_ids, image_tensor, image_sizes = next(data_loader_iter)
                
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids.to(device='cuda', non_blocking=True),
                        images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                        image_sizes=image_sizes,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True
                    )

                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                for index, output in zip(indices, outputs):
                    line = questions[index]
                    # idx = line["question_id"] if 'question_id' in line else line["id"]
                    cur_prompt = line["text"]
                    gt_reports = line["gt_reports"]  
                    study_id = line["study_id"]
                    dicom_id = line["dicom_id"]                  
                    ans_id = shortuuid.uuid()
                    ans_file.write(json.dumps({
                        'study_id': study_id,
                        'dicom_id': dicom_id,
                        'gt_reports': gt_reports,
                        "prompt": cur_prompt,
                        "caption": output.strip(),
                        "answer_id": ans_id,
                        "model_id": model_name
                    }) + "\n")
                ans_file.flush()
                pbar.update(1)                
            except StopIteration:
                # 如果迭代器结束，则退出循环
                break
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

        ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llama3")
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