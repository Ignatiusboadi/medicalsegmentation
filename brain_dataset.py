from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os


class BrainDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.image_folder = os.path.join(root_dir, "images")
        self.mask_folder = os.path.join(root_dir, "masks")
        self.image_files = sorted(os.listdir(self.image_folder))
        self.mask_files = sorted(os.listdir(self.mask_folder))
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        image_gray = np.array(image.convert("L")).astype(float)
        mask_name = self.mask_files[idx]
        mask_path = os.path.join(self.mask_folder, mask_name)
        mask = Image.open(mask_path).convert("L")
        if self.image_transform:
            image_gray = self.image_transform(image_gray)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image_gray, mask
