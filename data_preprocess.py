import os
import json
import cv2
import numpy as np
import skimage.draw
import tifffile
import shutil
import mlflow
import dagshub

dagshub.init(repo_owner='ignatiusboadi', repo_name='dagshub_proj_II', mlflow=True)

os.environ['MLFLOW_TRACKING_USERNAME'] = 'ignatiusboadi'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '67ea7e8b48b9a51dd1748b8bb71906cc5806eb09'
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/ignatiusboadi/dagshub_proj_II.mlflow'

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("proj-II-data-preprocessing")


def start_or_get_run():
    if mlflow.active_run() is None:
        mlflow.start_run()
    else:
        print(f"Active run with UUID {mlflow.active_run().info.run_id} already exists")


def end_active_run():
    if mlflow.active_run() is not None:
        mlflow.end_run()


def create_mask(image_info, annotations, output_folder):
    """function to create mask using coco annotations"""
    mask_np = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

    for ann in annotations:
        if image_info['id'] == ann['image_id']:
            for seg in ann['segmentation']:
                rr, cc = skimage.draw.polygon(seg[1::2], seg[0::2], mask_np.shape)
                mask_np[rr, cc] = 255  # Set the mask pixels to 255 for tumor areas

    mask_path = os.path.join(output_folder, f"{image_info['file_name'].replace('.jpg', '')}_mask.tif")
    tifffile.imwrite(mask_path, mask_np)


def process_images(json_file, mask_output_folder, image_output_folder, original_image_dir):
    """Take each image and create a mask with the location of the tumor."""
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


def compare_folders(folder1, folder2):
    """function compares folders and removes images without corresponding masks
     and masks without corresponding images"""
    f1_contents = os.listdir(folder1)
    f1_names = [file[:4] for file in f1_contents]
    f2_contents = os.listdir(folder2)
    f2_names = [file[:4] for file in f2_contents]
    f1_only = set(f1_names) - set(f2_names)  # images in folder 1 only
    f2_only = set(f2_names) - set(f1_names)  # images in folder 2 only

    for file in f1_contents:
        if file[:4] in f1_only:
            os.remove(os.path.join(folder1, file))
    for file in f2_contents:
        if file[:4] in f2_only:
            os.remove(os.path.join(folder2, file))


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

process_images(train_json_file, train_mask_output, train_image_output, f"{data_dir}/train")
process_images(test_json_file, test_mask_output, test_image_output, f"{data_dir}/test")
process_images(valid_json_file, val_mask_output, val_image_output, f"{data_dir}/valid")

mlflow.log_param('tr num before', len(os.listdir(train_image_output)))
mlflow.log_param('ts num before', len(os.listdir(test_image_output)))
mlflow.log_param('val num before', len(os.listdir(val_image_output)))

compare_folders(train_image_output, train_mask_output)
compare_folders(val_image_output, val_mask_output)
compare_folders(test_image_output, test_mask_output)

mlflow.log_param('tr num after', len(os.listdir(train_image_output)))
mlflow.log_param('ts num after', len(os.listdir(test_image_output)))
mlflow.log_param('val num after', len(os.listdir(val_image_output)))
