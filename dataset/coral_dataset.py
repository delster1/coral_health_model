import os
import random
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image
from icecream import ic
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, color
from skimage.segmentation import watershed
from skimage.io import imread
from utils.utils import generate_mask_and_label, get_coral_image, rgba2rgb_safe

class CoralDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = sorted(os.listdir(img_dir))  # Ensure matching order
        self.augment = augment

    def __getitem__(self, idx):
        # idx += 1
        # img = get_coral_image(self.img_dir, idx)

        img_path = os.path.join(self.img_dir, self.files[idx])
        mask_path = os.path.join(self.mask_dir, self.files[idx])

        ic(img_path)
        ic(mask_path)
        img = imread(img_path)

        # Load mask using skimage.io
        mask_np = imread(mask_path)

        if img.shape[-1] == 4:
            img = rgba2rgb_safe(img)
        if mask_np.shape[-1] == 4:
            mask_np = rgba2rgb_safe(mask_np)


        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1)

        ic("img stats:", img.min().item(), img.max().item(), img.mean().item())

        mask = torch.from_numpy(mask_np).long()

        if mask.ndim == 3:
            mask = mask.squeeze(-1)  # Remove channel dimension if present

        ic("img shape: ",  img.shape)
        ic("mask shape: ",mask.shape)

        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

            if random.random() > 0.5:
                angle = random.uniform(-30, 30)
                img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
                mask = mask.unsqueeze(0)  # Add dummy channel for rotation
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
                mask = mask.squeeze(0)    # Remove dummy channel after rotation

        return img, mask
    def __len__(self):
        return len(os.listdir(f"{self.img_dir}"))

# Create a dataset instance and data loader
dataset = CoralDataset('data/aug_images-flouro', 'data/aug_masks-flouro')
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

