from PIL import Image
from torch.utils.data import Dataset
import os
import torch


class ProdBrainDataset(Dataset):
    def __init__(self, root_dir, img_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_gray = img.convert('L')

        if self.transform:
            img_gray = self.transform(img_gray)

        return img_gray
    
    def get_original_size(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path)
        return img.size


class BrainDataset(ProdBrainDataset):
    def __init__(self, root_dir, img_dir, mask_dir=None, transform=None):
        super().__init__(root_dir, img_dir, transform)
        self.mask_dir = mask_dir
        if mask_dir:
            self.mask_files = [f for f in os.listdir(self.mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]


    # def __getitem__(self, idx):
    
    #     img_gray = super().__getitem__(idx)
    #     if self.mask_dir:
            
    #         if idx >= len(self.mask_files):
    #             raise IndexError(f"Index {idx} is out of range for mask_files with length {len(self.mask_files)}.")
    #         mask_name = self.mask_files[idx]
    #         mask_path = os.path.join(self.mask_dir, mask_name)
    #         if not os.path.exists(mask_path):
    #             raise FileNotFoundError(f"Mask file '{mask_path}' does not exist.")
    #         mask = Image.open(mask_path).convert('L')
    #         if self.transform:
    #             mask = self.transform(mask)
    #         return img_gray, mask
    #     return img_gray, None

    def __getitem__(self, idx):
        img_gray = super().__getitem__(idx)
        
        if self.mask_dir and len(self.mask_files) > idx:
            mask_name = self.mask_files[idx]
            mask_path = os.path.join(self.mask_dir, mask_name)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file '{mask_path}' does not exist.")
            mask = Image.open(mask_path).convert('L')
            if self.transform:
                mask = self.transform(mask)
            return img_gray, mask
    
        return img_gray, torch.zeros_like(img_gray)#None

