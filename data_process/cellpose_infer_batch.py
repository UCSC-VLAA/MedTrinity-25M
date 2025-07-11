import os
import numpy as np
from cellpose import models, io
from cellpose.io import imread
from PIL import Image, ImageDraw
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from scipy.ndimage import label
import json

io.logger_setup()

def get_bounding_boxes_and_save_wmask(mask, image_file, wmask_file):
    # Find all non-zero regions in the mask
    labeled, num_features = label(mask)
    # image = Image.open(image_file)
    # draw = ImageDraw.Draw(image)
    bboxes = []
    for feature in range(1, 3):
        # Get coordinates of the feature
        coords = np.argwhere(labeled == feature)
        # Determine the bounding box
        top_left = coords.min(axis=0)
        bottom_right = coords.max(axis=0)
        bbox = [top_left[1], top_left[0], bottom_right[1], bottom_right[0]]
        # draw.rectangle(bbox, outline="green", width=2)
        bboxes.append([int(i) for i in bbox])
        
    # image.save(wmask_file)
    return bboxes

def norm01(arr):
    # norm the image mask to the binary mask
    norm01_array = np.zeros(arr.shape)
    norm01_array[arr > 0] = 255
    return norm01_array.astype(np.uint8)

def get_all_images(root_dir):
    files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')) and 'mask' not in filename:
                files.append(os.path.join(dirpath, filename))
    return files

def save_masks(files, masks, mask_path, wmask_path):
    mask_message = []
    for f, m in zip(files, masks):
        mask_image = Image.fromarray(norm01(m))
        file_name = os.path.basename(f)
        base = os.path.splitext(file_name)[0]
        mask_file = os.path.join(mask_path, base + '_mask.jpg')
        wmask_file = os.path.join(wmask_path, base + '_wmask.jpg')
        # mask_image.save(mask_file)
        bboxes = get_bounding_boxes_and_save_wmask(m, f, wmask_file)
        mask_message.append({'image_path': f, 'bboxes': bboxes, 'mechine': args.mechine_name})
    return mask_message

def cellpose_infer_batch(args, i):
    image_files = [f for idx, f in enumerate(args.image_files) if idx % args.num_gpus == i]
    model = models.Cellpose(model_type=args.model_type, gpu=args.use_gpu, device=i+2)
    channels = [[0, 0]]
    mask_message = []
    nimg = len(image_files)
    for batch_start in tqdm(range(0, nimg, args.batch_size), total=nimg//args.batch_size, desc=f'GPU {i} Processing Cellpose'):
        batch_end = min(batch_start + args.batch_size, nimg)
        batch_files = image_files[batch_start:batch_end]
        batch_images = [imread(f) for f in batch_files]
        masks, _, _, _ = model.eval(batch_images, batch_size=args.batch_size, diameter=args.diameter, channels=channels, cellprob_threshold=args.cellprob_threshold)
        mask_message.extend(save_masks(batch_files, masks, args.mask_path, args.wmask_path))
    return mask_message

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='get the cellpose batch inference args')
    
    
    parser.add_argument('--image_path', type=str, default='.', help='path to the image files')
    parser.add_argument('--model_type', type=str, default='cyto3', help='model type')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('--num_gpus', type=int, default=8, help='number of gpus to use')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for inference')
    parser.add_argument('--diameter', type=float, default=30.0, help='diameter of the cells')
    parser.add_argument('--cellprob_threshold', type=float, default=0.0, help='cell probability threshold')
    parser.add_argument('--mask_path', type=str, default='.', help='path to save the output masks')
    parser.add_argument('--wmask_path', type=str, default='.', help='path to save the output wmasks')
    parser.add_argument('--mechine_name', type=str, default='2u2', help='mechine name')
    parser.add_argument('--mask_json', type=str, default='mask_message.json', help='json file to save the mask message')

    args = parser.parse_args()
    
    args.image_files = get_all_images(args.image_path)
    # os.makedirs(args.mask_path, exist_ok=True)
    # os.makedirs(args.wmask_path, exist_ok=True)   
    with ThreadPoolExecutor(max_workers=args.num_gpus) as executor:
        futures = []
        for i in range(args.num_gpus):
            futures.append(executor.submit(cellpose_infer_batch, args, i))

        message_list = []
        
        for future in as_completed(futures):
            # try: 
            message_list.extend(future.result())
            # except Exception as e:
            #     print(f"Error in future: {e}")
            
    with open(args.mask_json, 'w') as f:
        json.dump(message_list, f)
