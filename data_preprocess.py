import os
import json
import cv2
import numpy as np
import shutil
import mlflow
import dagshub
from PIL import Image


os.environ['MLFLOW_TRACKING_USERNAME'] = 'ignatiusboadi'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '67ea7e8b48b9a51dd1748b8bb71906cc5806eb09'
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/ignatiusboadi/dagshub_proj_II.mlflow'

dagshub.init(repo_owner='ignatiusboadi', repo_name='dagshub_proj_II', mlflow=True)

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


with open('train_annotations.coco.json', 'r') as train_file:
    train_annotations = json.load(train_file)

with open("test_annotations.coco.json", "r") as test_file:
    test_annotations = json.load(test_file)

with open("valid_annotations.coco.json", "r") as valid_file:
    valid_annotations = json.load(valid_file)


def create_mask(image_path, data, output_dir):
    """function to create mask using coco annotations"""
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    file_name = os.path.basename(image_path)
    image_id = None
    width, height = None, None

    for image_info in data['images']:
        if image_info['file_name'] == file_name:
            image_id = image_info['id']
            width = image_info['width']
            height = image_info['height']
            break

    if image_id is None:
        print(f"Image {file_name} not found in dataset.")
        return

    mask = np.zeros((height, width), dtype=np.uint8)

    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            for segmentation in annotation['segmentation']:
                polygon = np.array(segmentation).reshape(-1, 2)
                cv2.fillPoly(mask, [polygon.astype(np.int32)], color=1)

    mask_path = os.path.join(output_dir, file_name)
    Image.fromarray(mask * 255).save(mask_path)


def get_all_mask_imgs(data, mask_output_dir, img_output_dir, origin_img_dir):
    """Take each image and create a mask with the location of the tumor."""
    images = data['images']

    if not os.path.exists(mask_output_dir):
        os.makedirs(mask_output_dir)
    if not os.path.exists(img_output_dir):
        os.makedirs(img_output_dir)

    for img in images:
        img_path = os.path.join(origin_img_dir, img['file_name'])
        create_mask(img_path, data, mask_output_dir)
        origin_img_path = os.path.join(origin_img_dir, img['file_name'])
        new_img_path = os.path.join(img_output_dir, os.path.basename(origin_img_path))
        shutil.copy2(origin_img_path, new_img_path)


def compare_folders(folder1, folder2):
    """function compares folders and removes images without corresponding masks
     and masks without corresponding images"""
    f1_contents = os.listdir(folder1)
    f1_names = [file[:4] for file in f1_contents]
    f2_contents = os.listdir(folder2)
    f2_names = [file[:4] for file in f2_contents]
    f1_only = set(f1_names) - set(f2_names)
    f2_only = set(f2_names) - set(f1_names)

    for file in f1_contents:
        if file[:4] in f1_only:
            os.remove(os.path.join(folder1, file))
    for file in f2_contents:
        if file[:4] in f2_only:
            os.remove(os.path.join(folder2, file))


train_image_output = 'images/train'
test_image_output = 'images/test'
val_image_output = 'images/valid'
train_mask_output = 'masks/train'
test_mask_output = 'masks/test'
val_mask_output = 'masks/valid'

get_all_mask_imgs(train_annotations, 'masks/train', 'images/train', 'brain_data/train')
get_all_mask_imgs(valid_annotations, 'masks/valid', 'images/valid', 'brain_data/valid')
get_all_mask_imgs(test_annotations, 'masks/test', 'images/test', 'brain_data/test')


mlflow.log_param('tr num before', len(os.listdir(train_image_output)))
mlflow.log_param('ts num before', len(os.listdir(test_image_output)))
mlflow.log_param('val num before', len(os.listdir(val_image_output)))

compare_folders(train_image_output, train_mask_output)
compare_folders(val_image_output, val_mask_output)
compare_folders(test_image_output, test_mask_output)

mlflow.log_param('tr num after', len(os.listdir(train_image_output)))
mlflow.log_param('ts num after', len(os.listdir(test_image_output)))
mlflow.log_param('val num after', len(os.listdir(val_image_output)))
