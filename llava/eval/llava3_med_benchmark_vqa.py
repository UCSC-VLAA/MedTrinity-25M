from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from transformers import CLIPImageProcessor, GenerationConfig
# from llava import LlavaLlamaForCausalLM
# from llava.model.utils import KeywordsStoppingCriteria
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from tqdm import tqdm
from multimedeval import MultiMedEval, EvalParams, SetupParams
import json
import logging

logging.basicConfig(level=logging.INFO)

# DEFAULT_IMAGE_TOKEN = "<image>"
# DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
# DEFAULT_IM_START_TOKEN = "<im_start>"
# DEFAULT_IM_END_TOKEN = "<im_end>"

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class batcherLLaVA_Med:
    def __init__(self, args):
        disable_torch_init()
        self.args = args
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(model_path, None, model_name, use_flash_attn=True)

        # self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        # self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        self.model.config.tokenizer_padding_side = self.tokenizer.padding_side = "left"

        # Set the model's padding token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.conv_mode = args.conv_mode

        # vision_tower = self.model.model.vision_tower[0]
        # vision_tower.to(device="cuda", dtype=torch.float16)
        # vision_config = vision_tower.config
        # vision_config.im_patch_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        # vision_config.use_im_start_end = True
        # vision_config.im_start_token, vision_config.im_end_token = self.tokenizer.convert_tokens_to_ids(
        #     [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
        # )

        # self.image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
        # self.image_processor = CLIPImageProcessor.from_pretrained(
        #     self.model.config.mm_vision_tower, torch_dtype=torch.float16
        # )

        # self.generation_config = GenerationConfig(
        #     eos_token_id=self.tokenizer.eos_token_id,
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     bos_token_id=self.tokenizer.bos_token_id,
        # )

    def __call__(self, prompts):
        outputList = []
        listText = []
        listImage = []
        listImageSize = []
        for prompt in prompts:
            conv = conv_templates[self.conv_mode].copy()
            for message in prompt[0]:
                qs: str = message["content"]
                print(qs)
                qs = qs.replace("<image>", "").strip()
                if self.model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            textPrompt = conv.get_prompt()

            listText.append(textPrompt)

            for image in prompt[1]:
                image = image.convert("RGB")
                listImageSize.append(image.size)
                image_tensor = process_images([image], self.image_processor, self.model.config)[0]

                listImage.append(image_tensor)

        inputs_ids = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) for prompt in listText]
        
        input_ids = self.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        
        image_tensor = torch.stack(listImage, dim=0) if len(listImage) > 0 else None
        
        image_sizes = [tuple(size) for size in listImageSize]
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids.to(device='cuda', non_blocking=True),,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=self.args.temperature > 0,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                num_beams=self.args.num_beams,
                max_new_tokens=self.args.max_new_tokens,
                use_cache=True
            )
            
        outputs_batch = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # Measure time spent
        for outputs in outputs_batch:
            while True:
                cur_len = len(outputs)
                outputs = outputs.strip()
                for pattern in ["###", "Assistant:", "Response:"]:
                    if outputs.startswith(pattern):
                        outputs = outputs[len(pattern) :].strip()
                if len(outputs) == cur_len:
                    break

            try:
                index = outputs.index(conv.sep)
            except ValueError:
                outputs += conv.sep
                index = outputs.index(conv.sep)

            outputs = outputs[:index].strip()
            outputList.append(outputs)

        return outputList
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    # parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-folder", type=str, default="")
    # parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    # parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llama3")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    batcher = batcherLLaVA_Med(
        args=args
    )

    physionet_username = os.getenv("PHYSIONET_USERNAME")
    physionet_password = os.getenv("PHYSIONET_PASSWORD")
    engine = MultiMedEval()
    setupParams = SetupParams(MIMIC_CXR_dir="/data1/yunfei/EvalData/MIMIC_CXR",MIMIC_III_dir="/data1/yunfei/EvalData/MIMIC_III", MedNLI_dir="/data1/yunfei/EvalData/MedNLI", physionet_username=physionet_username, physionet_password=physionet_password)
    # setupParams = SetupParams(MedNLI_dir="/data1/yunfei/EvalData/MedNLI")
    try: 
        engine.setup(setupParams)
    except:
        print("[Error]: engine setup fail!")
    engine.eval(["MedNLI"], batcher, EvalParams(batch_size=2, run_name="testLLaVAMed3"))
