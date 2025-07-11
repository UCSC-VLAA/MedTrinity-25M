import os
import torch
from datasets import load_dataset, Dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

from llava.conversation import conv_templates
import json

@dataclass
class Arguments(transformers.TrainingArguments):
    data_file: str = field(
        default="metadata.jsonl",
        metadata={"help": "The jsonl file path of data."}
    )
    model_path: str = field(
        default="./Llama-3-8B-Instruct",
        metadata={"help": "The model need to finetune."}
    )
    new_model: str = field(
        default="Llama-3-8B-Instruct-reformat",
        metadata={"help": "The finetuned model's name."}
    )
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(default="./results")
    max_seq_length: int = field(
        default=8192,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    packing: bool = field(
        default=False,
        metadata={
            "help":
                "Pack multiple short examples in the same input sequence to increase efficiency."
        },
    )


def QA2Text(example):
    cap = example["caption"]
    ans = example["answer"]
    conv = conv_templates["llama3_qa"].copy()
    conv.append_message(conv.roles[0], cap)
    conv.append_message(conv.roles[1], ans)
    prompt = conv.get_prompt()
    example["text"] = prompt.replace("<image>\n", "")
    
    return example
    
    
def train():
    parser = transformers.HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Load dataset and convert the captions & answers to texts
    dataset = load_dataset("json", data_files=args.data_file, split="train")
    # updated_dataset = dataset.map(QA2Text, remove_columns=["caption", "answer", "image", "source"], batched=True, batch_size=16,)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        attn_implementation="flash_attention_2", 
        torch_dtype=(torch.bfloat16 if args.bf16 else None),
        cache_dir=args.cache_dir
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # Load LoRA configuration
    # peft_config = LoraConfig(
    #     lora_alpha=lora_alpha,
    #     lora_dropout=lora_dropout,
    #     r=lora_r,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
    
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to=args.report_to
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=None,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=args.packing,
    )

    # Train model
    trainer.train()

    # Save trained model
    trainer.model.save_pretrained(args.new_model)
    

if __name__ == "__main__":    
    train()