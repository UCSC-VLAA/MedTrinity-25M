import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
import json
    
    
SYSTEM_PROMPT = '''
"You are an AI assistant specialized in biomedical topics."
    You are now provided with a fine-grained caption about a medical imaging image, including the Modality, Organ & Tissue 
    Detection, ROI Location & Description, Disease-related Color & Texture, and Region Relationship of this medical imaging image,
    but you cannot contact the real medical image. Please use the provided fine-grained caption to propose high-quality visual 
    question answer (VQA) questions and answers for Modality, Organ & Tissue Detection, ROI Location & Description, Disease-related
    Color & Texture, and Region Relationship information of medical imaging images. The questions and answers produced need to meet
    the following requirements:
1. Leverage relevant medical knowledge you know to give high-quality VQA questions and answers;
2. Make sure the questions can deduce the answers, the questions and answers are logical, and the answers can be found in the provided fine-grained caption;
3. On the basis of the above two requirements, ensure the diversity of the questions. The provided VQA example is a simple one; try to enrich the variety of question styles as much as possible, such as generating multiple-choice questions with options 1, 2, 3, 4, or A, B, C, D, or close-ended yes-or-no questions;
4.Please create VQA in the format of the example:"<q>question</q>,<a>answer </a>".
'''    

def main(args):
    if not os.path.exists(args.caption_file):
        print("The caption file does not exist.")
        return
    
    with open(args.caption_file, 'r') as f:
        metadata = [json.loads(line) for line in f]
        
    os.makedirs(os.path.dirname(args.reformat_file), exist_ok=True)

    with open(args.reformat_file, "w") as ref_file:
        for i, d in enumerate(metadata):
            messages = [{"role": "system", "content": SYSTEM_PROMPT},{"role": "user", "content": f"Here is caption: {d['caption']}"}]
            body = {
                "model": args.model,
                "messages": messages,
                "max_tokens": args.max_tokens,
            }
            ref_file.write(
                json.dumps({
                    "custom_id": d["id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }) + '\n'
            )   
            # test
            if i > 1000: break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reformat the original jsonl format to the openai batch call's format")

    parser.add_argument("--caption-file", type=str, default="metadata.jsonl")
    parser.add_argument("--reformat-file", type=str, default="batchinput.jsonl")
    parser.add_argument('--model', type=str, default="gpt-4o", help='the name of openai model')
    parser.add_argument("--max_tokens", type=int, default=4096)
    
    args = parser.parse_args()
    
    main(args)