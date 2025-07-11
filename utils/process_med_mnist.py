import os
import json
from medmnist.dataset import OCTMNIST, PathMNIST, PneumoniaMNIST, RetinaMNIST, BloodMNIST, ChestMNIST, OrganAMNIST, OrganCMNIST, DermaMNIST, BreastMNIST, TissueMNIST, OrganSMNIST, MedMNIST2D
import concurrent.futures
from tqdm import tqdm
from PIL import Image

def getCorrectAnswer(options, sample, fullText=False) -> int:
    label = sample[1].tolist()
    
    if fullText:
        return ",".join([options[str(l + 1)] for l in label])
    
    if len(label) == 1:
        label = label[0]
    
    return label
    
def format_vqa(options, sample):
    question = "<img> Options:\n"
    question += " \n ".join([f"{option}: {options[option]}" for option in options])
    question += " \n Which options correspond to the image?"

    formattedText = [
        {
            "from": "human",
            "value": question,
        }
    ]
    
    formattedText.append({"from": "gpt", "value": f"{getCorrectAnswer(options, sample, fullText=True)}"})

    return formattedText

def process_sample(sample, idx, mnist_name, options, modality, cachedirName):
    formattedText = format_vqa(options, sample)

    img_path = os.path.join(cachedirName, "images", f"{mnist_name}_{idx}.jpg")
    sample[0].save(img_path)
    return {
        "id": f"{mnist_name}_{idx}",
        "image": f"{mnist_name}_{idx}.jpg",
        "modality": modality,
        "conversations": formattedText
    }

def process_dataset(mnist_name, cachedirName):
    dataset_class = NAME_TO_MNIST[mnist_name]["class"]
    modality = NAME_TO_MNIST[mnist_name]["modality"]
    dataset = dataset_class(split="train", download=True, root=cachedirName)
    options = {str(int(key) + 1): value for key, value in dataset.info["label"].items()}
    
    results = []
    progress_bar = tqdm(total=len(dataset), desc=f'Processing {mnist_name} ...')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_sample = {executor.submit(process_sample, dataset[idx], idx, mnist_name, options, modality, cachedirName): idx for idx in range(len(dataset))}
        for future in concurrent.futures.as_completed(future_to_sample):
            try:
                result = future.result()
                results.append(result)
                progress_bar.update(1)
            except Exception as exc:
                idx = future_to_sample[future]
                print(f'Sample {idx} generated an exception: {exc}')
    
    return results

cachedirName = "/home/ec2-user/disk/llava_med/Data/Med_MNIST"
os.makedirs(cachedirName, exist_ok=True)
os.makedirs(os.path.join(cachedirName,"images"), exist_ok=True)

NAME_TO_MNIST = {
    "OCTMNIST": {"class": OCTMNIST, "modality": "OCT" },
    "PathMNIST": {"class": PathMNIST, "modality": "Pathology" },
    "PneumoniaMNIST": {"class": PneumoniaMNIST, "modality": "X-Ray" },
    "RetinaMNIST": {"class": RetinaMNIST, "modality": "Fundus Camera" },
    "BloodMNIST": {"class": BloodMNIST, "modality": "Microscope" },
    "ChestMNIST": {"class": ChestMNIST, "modality": "X-Ray" },
    "OrganAMNIST": {"class": OrganAMNIST, "modality": "CT" },
    "OrganCMNIST": {"class": OrganCMNIST, "modality": "CT" },
    "OrganSMNIST": {"class": OrganSMNIST, "modality": "CT" },
    "DermaMNIST": {"class": DermaMNIST, "modality": "Dermatology" },
    "BreastMNIST": {"class": BreastMNIST, "modality": "Ultrasound" },
    "TissueMNIST": {"class": TissueMNIST, "modality": "Microscope" },
}

mnist_name_list = ["OCTMNIST", "PathMNIST", "PneumoniaMNIST", "RetinaMNIST", "BloodMNIST", "ChestMNIST", "OrganAMNIST", "OrganCMNIST", "OrganSMNIST", "DermaMNIST", "BreastMNIST", "TissueMNIST"]

train_list = []

# def process_all_datasets(mnist_name_list, cachedirName):
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         future_to_mnist = {executor.submit(process_dataset, mnist_name, cachedirName): mnist_name for mnist_name in mnist_name_list}
        
#         for future in concurrent.futures.as_completed(future_to_mnist):
#             mnist_name = future_to_mnist[future]
#             try:
#                 data = future.result()
#                 train_list.extend(data)
#             except Exception as exc:
#                 print(f'{mnist_name} generated an exception: {exc}')

# process_all_datasets(mnist_name_list, cachedirName)

# with open(os.path.join(cachedirName, "train.json"), "w") as f:
#     json.dump(train_list, f)
for mnist_name in mnist_name_list: 
    results = process_dataset(mnist_name, cachedirName)
    train_list.extend(results)
    
with open(os.path.join(cachedirName, "train.json"), "w", encoding='utf-8') as f:
    json.dump(train_list, f, ensure_ascii=False, indent=4)
