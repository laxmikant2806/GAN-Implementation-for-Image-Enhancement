import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset
from random import randint
import numpy as np

class PairedImageDataset(Dataset):
    """Optimized Paired HR/LR patches with train/val split on-the-fly."""
    def __init__(self, data_dir: Path, upscale: int, patch: int, split='train', train_ratio=0.8, cache_size=1000):
        # Gather all image files (supports multiple formats)
        self.hr_files = sorted(list(data_dir.glob("*.png")) + 
                              list(data_dir.glob("*.jpg")) + 
                              list(data_dir.glob("*.jpeg")))

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
        
        # Simple caching mechanism for frequently accessed images
        self.cache = {}
        self.cache_size = cache_size

    def _load_and_process_image(self, idx):
        """Load and process image with caching"""
        if idx in self.cache:
            return self.cache[idx]
        
        # Load image
        img_path = str(self.hr_files[idx])
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert to consistent grayscale format
        if img.ndim == 2:
            img = img[..., None]
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)[..., None]
        elif img.shape[2] == 3:  # RGB/BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
        
        # Cache if there's room
        if len(self.cache) < self.cache_size:
            self.cache[idx] = img
        
        return img

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_img = self._load_and_process_image(idx)
        h, w, _ = hr_img.shape

        # Ensure image is large enough for cropping
        if h < self.patch_hr or w < self.patch_hr:
            # Resize if too small
            scale_factor = max(self.patch_hr / h, self.patch_hr / w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            hr_img = cv2.resize(hr_img, (new_w, new_h))
            h, w, _ = hr_img.shape

        # Random crop from HR image
        top = randint(0, h - self.patch_hr)
        left = randint(0, w - self.patch_hr)
        hr_crop = hr_img[top:top+self.patch_hr, left:left+self.patch_hr, :]
        
        # Create LR version by downsampling
        lr_crop = cv2.resize(hr_crop, (self.patch_lr, self.patch_lr), interpolation=cv2.INTER_CUBIC)
        
        # Ensure proper dimensions
        if lr_crop.ndim == 2:
            lr_crop = lr_crop[..., None]

        # Convert to tensors (optimized)
        hr_tensor = torch.from_numpy(hr_crop).permute(2, 0, 1).float() / 255.0
        lr_tensor = torch.from_numpy(lr_crop).permute(2, 0, 1).float() / 255.0

        return lr_tensor, hr_tensor
