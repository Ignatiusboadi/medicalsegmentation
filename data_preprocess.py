import os
import json
import cv2
import numpy as np
import skimage.draw
import tifffile
import shutil


def create_mask(image_info, annotations, output_folder):
    mask_np = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

    for ann in annotations:
        if image_info['id'] == ann['image_id']:
            for seg in ann['segmentation']:
                rr, cc = skimage.draw.polygon(seg[1::2], seg[0::2], mask_np.shape)
                mask_np[rr, cc] = 255  # Set the mask pixels to 255 for tumor areas

    mask_path = os.path.join(output_folder, f"{image_info['file_name'].replace('.jpg', '')}_mask.tif")
    tifffile.imwrite(mask_path, mask_np)


def process_images(json_file, mask_output_folder, image_output_folder, original_image_dir):
    # Load the JSON file with the annotations
    with open(json_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']

    os.makedirs(mask_output_folder, exist_ok=True)
    os.makedirs(image_output_folder, exist_ok=True)

    for img in images:
        create_mask(img, annotations, mask_output_folder)
        original_image_path = os.path.join(original_image_dir, img['file_name'])
        new_image_path = os.path.join(image_output_folder, os.path.basename(original_image_path))
        shutil.copy2(original_image_path, new_image_path)


data_dir = 'brain_data'
processed_images_dir = 'processed_data'

train_json_file = 'train_annotations.coco.json'
train_mask_output = f'{processed_images_dir}/train/masks'
train_image_output = f'{processed_images_dir}/train/images'

test_json_file = 'test_annotations.coco.json'
test_mask_output = f'{processed_images_dir}/test/masks'
test_image_output = f'{processed_images_dir}/test/images'

valid_json_file = 'valid_annotations.coco.json'
val_mask_output = f'{processed_images_dir}/valid/masks'
val_image_output = f'{processed_images_dir}/valid/images'

# Run the processing function
process_images(train_json_file, train_mask_output, train_image_output, f"{data_dir}/train")
process_images(test_json_file, test_mask_output, test_image_output, f"{data_dir}/test")
process_images(valid_json_file, val_mask_output, val_image_output, f"{data_dir}/valid")
