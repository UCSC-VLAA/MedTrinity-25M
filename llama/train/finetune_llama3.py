from dataclasses import dataclass, field
import json
import math
import logging
import os
import copy
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, GPTQConfig, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from transformers import BitsAndBytesConfig

from llava import conversation as conversation_lib
from llava.conversation import conv_templates

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
SYSTEM_PROMPT = '''
You are an AI assistant specialized in biomedical topics. Please create VQA in the format of the example:"<q>question</q><a>answer</a >".\n
You are provided with a fine-grained caption of a medical image, including the Modality, Organ & Tissue 
Detection, ROI Location & Description, Disease-related Color & Texture, and Region Relationship of this medical image. Unfortunately, you don't have access to the actual image.
Below are requirements for generating the questions and answers in the conversation:\n
- Avoid quoting or referring to specific facts, terms, abbreviations, dates, numbers, or names, as these may reveal the conversation is based on the text information, rather than the image itself. Focus on the visual aspects of the image that can be inferred without the text information.\n
- Do not use phrases like "mentioned", "caption", "context" in the conversation. Instead, refer to the information as being "in the image."\n
- Ensure that questions are diverse and cover a range of visual aspects of the image.\n
- The conversation should include at least 2-3 turns of questions and answers about the visual aspects of the image.\n
- For general questions that start with "Do" or "is" or "are", please answer with "yes" or "no".\n
- For wh-questions that start with like 'what', please answer with a short phrase consisting of a few words.\n
- Answer responsibly, avoiding overconfidence, and do not provide medical advice or diagnostic information. Encourage the user to consult a healthcare professional for advice.
Below is the fine-grained need to be converted into VQA questions and answers in the format of the example:"<q>question</q><a>answer</a >". \n
'''


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="./Llama-3-8B-Instruct")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # ['gate_proj', 'o_proj', 'k_proj', 'q_proj', 'up_proj', 'down_proj', 'v_proj']
    lora_target_modules: List[str] = field(
        default_factory=lambda: ['o_proj', 'k_proj', 'q_proj', 'v_proj']
    )
    # lora_target_modules = None
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)
        
        
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    print("tokenizer's pad token id is: ", tokenizer.pad_token_id)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
    

def preprocess(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conv_templates["llama3_qa"].copy()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    convs, masks = [], []
    
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            print(f"Skipping the first one if it is not from human: {i}")
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        prompt = conv.get_prompt()
        convs.append(prompt)
        masks.append(prompt.split(roles[1])[0] + roles[1])

    return dict(
        convs=convs,
        masks=masks,
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.convs = data_dict["convs"]
        self.masks = data_dict["masks"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            convs=self.convs[i],
            masks=self.masks[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = dict(
            convs=ret["convs"][0],
            masks=ret["masks"][0],
        )
        self.cached_data_dict[i] = ret

        return ret
    
    
@dataclass
class DataCollatorForDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        convs, masks = tuple([instance[key] for instance in instances] for key in ("convs", "masks"))
        
        input_ids = tokenizer(
            convs,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        labels = copy.deepcopy(input_ids)
        
        mask_ids = tokenizer(
            masks,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids.ne(tokenizer.pad_token_id)
        
        pads = torch.full((mask_ids.shape[0], labels.shape[1]-mask_ids.shape[1]), False)
        mask_ids = torch.cat((mask_ids, pads), dim=1)
        
        labels[mask_ids] = IGNORE_TOKEN_ID
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    # train_json = json.load(open(data_args.data_path, "r"))
    if data_args.data_path.endswith(".jsonl"):
        with open(data_args.data_path, "r") as f:
            train_json = [json.loads(line) for line in f]
    elif data_args.data_path.endswith(".json"):
        train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        # eval_json = json.load(open(data_args.eval_data_path, "r"))
        if data_args.eval_data_path.endswith(".jsonl"):
            with open(data_args.eval_data_path, "r") as f:
                eval_json = [json.loads(line) for line in f]
        elif data_args.eval_data_path.endswith(".json"):
            eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None
        
    data_collator = DataCollatorForDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def get_quantization_config(model_args):
    if model_args.load_in_4bit:
        compute_dtype = torch.float16
        # if model_args.torch_dtype not in {"auto", None}:
        #     compute_dtype = getattr(torch, model_args.torch_dtype)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1)) == 1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are incompatible with QLoRA."
            )

    is_chat_model = 'instruct' in model_args.model_name_or_path.lower()
    if (
            training_args.use_lora
            and not lora_args.q_lora
            and deepspeed.is_deepspeed_zero3_enabled()
            and not is_chat_model
    ):
        raise RuntimeError("ZeRO3 is incompatible with LoRA when finetuning on base model.")

    model_load_kwargs = {
        'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
    }

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # Load model and tokenizer
    quantization_config = get_quantization_config(lora_args)

    rank0_print("quantization_config:", quantization_config)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=quantization_config if lora_args.q_lora else None,
        **model_load_kwargs,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if training_args.use_lora:
        if is_chat_model:
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]

        def find_all_linear_names(args, model):
            import bitsandbytes as bnb
            cls = bnb.nn.Linear4bit if args.load_in_4bit == 4 else (
                bnb.nn.Linear8bitLt if args.load_in_8bit == 8 else torch.nn.Linear)
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    names = name.split('.')
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])

            if 'lm_head' in lora_module_names:  # needed for 16-bit
                lora_module_names.remove('lm_head')
            return list(lora_module_names)

        if lora_args.lora_target_modules is None:
            lora_args.lora_target_modules = find_all_linear_names(lora_args, model)

        print(lora_args.lora_target_modules)
        print(modules_to_save)

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()


    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    with torch.autocast("cuda"):
        trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()