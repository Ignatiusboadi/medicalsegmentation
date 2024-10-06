from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os


class BrainDataset(Dataset):
    def __init__(self, root_dir, img_dir, mask_dir=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = [f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if mask_dir:
            self.mask_files = [f for f in os.listdir(self.mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_gray = img.convert('L')

        if self.mask_dir:
            mask_name = self.mask_files[idx]
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = Image.open(mask_path).convert('L')

        if self.transform:
            img_gray = self.transform(img_gray)
            if self.mask_dir:
                mask = self.transform(mask)
                output = img_gray, mask
            else:
                output = img_gray, None

        return output
