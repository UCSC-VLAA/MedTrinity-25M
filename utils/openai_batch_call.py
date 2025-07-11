from openai import AzureOpenAI, OpenAI
import json
import argparse
import os
import time


def format_elapsed_time(elapsed_time):
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    return f"{hours:.2f} : {minutes:.2f} : {seconds:.2f}"


def sigle_client_batch_call(client, batch_input, response_file):

    batch_input_file = client.files.create(
    file=open(batch_input, "rb"),
    purpose="batch"
    )
    
    batch_input_file_id = batch_input_file.id
    
    start_time = time.time()
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": "medical data reformat"
        }
    )
    
    interval_time = 10
    
    os.makedirs(os.path.dirname(response_file), exist_ok=True)
    
    while True:
        time.sleep(interval_time)
        batch = client.batches.retrieve(batch_input_file_id)
        elapsed_time = format_elapsed_time(interval_time)
        if batch.status == "completed":
            print("GPT reformat caption done!")
            file_response = client.files.content(batch.output_file_id)
            with open(response_file, "w") as f:
                json.dump(list(file_response.text), f, indent=4)
        elif batch.status in ["failed","expired","cancelling","cancelled"]:
            print(f"Error: the batch status is {bathc.status}.")
            break
        elif batch.status == "in_progress":
            print(f"[Request Counts]: completed {batch.request_counts.completed}, failed {batch.request_counts.failed}, total {batch.request_counts.total} || [Time]: {elapsed_time}<{format_elapsed_time(interval_time * batch.request_counts.total / batch.request_counts.completed)}, {interval_time / batch.request_counts.completed}s/it")
        else:
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Call openai batch inference.')

    parser.add_argument("--batch_input", type=str, default="batchinput.jsonl")
    parser.add_argument("--response_file", type=str, default="response.jsonl")
    
    args = parser.parse_args()
    
    client = OpenAI(
        organization='org-5fz09SUguUCh5xbxXn9cFVEw',
        project='proj_SUOvYoowmhzCm5hP2mbkXjOL',
    )
    
    sigle_client_batch_call(client, args.batch_input, args.response_file)
