import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset
from random import randint
import numpy as np

class PairedImageDataset(Dataset):
    """Paired HR/LR patches with train/val split on-the-fly."""
    def __init__(self, data_dir: Path, upscale: int, patch: int, split='train', train_ratio=0.8):
        # Gather all .png images
        self.hr_files = sorted(list(data_dir.glob("*.png")))

        # Compute split index
        total_images = len(self.hr_files)
        split_idx = int(total_images * train_ratio)
        if split == 'train':
            self.hr_files = self.hr_files[:split_idx]
        elif split == 'val':
            self.hr_files = self.hr_files[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.upscale = upscale
        self.patch_hr = patch
        self.patch_lr = patch // upscale

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_img = cv2.imread(str(self.hr_files[idx]), cv2.IMREAD_UNCHANGED)

        # ---- Force all images to be grayscale ----
        if hr_img.ndim == 2:
            # Already grayscale
            img = hr_img
        elif hr_img.shape[2] == 4:
            # Convert RGBA to grayscale
            img = cv2.cvtColor(hr_img, cv2.COLOR_BGRA2GRAY)
        else:
            # Convert RGB to grayscale
            img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2GRAY)

        # Add channel dimension so shape is (H, W, 1)
        img = img[..., None]

        h, w, _ = img.shape

        # Random crop
        top  = randint(0, h - self.patch_hr)
        left = randint(0, w - self.patch_hr)
        hr_crop = img[top:top+self.patch_hr, left:left+self.patch_hr, :]

        # Downsample for low-resolution
        lr_crop = cv2.resize(hr_crop, (self.patch_lr, self.patch_lr), interpolation=cv2.INTER_CUBIC)

        # Ensure LR patch has channel dim
        if lr_crop.ndim == 2:
            lr_crop = lr_crop[..., None]

        # Convert to tensor
        hr_tensor = torch.from_numpy(hr_crop.transpose(2, 0, 1)).float() / 255.
        lr_tensor = torch.from_numpy(lr_crop.transpose(2, 0, 1)).float() / 255.
        return lr_tensor, hr_tensor
