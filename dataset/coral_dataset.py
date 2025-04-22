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
from utils.utils import get_mask, get_coral_image, rgba2rgb_safe


class CoralDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=False, grayscale=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = sorted(os.listdir(img_dir))  # Ensure matching order
        self.augment = augment
        self.grayscale = grayscale

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        # ic(img_path)

        if self.grayscale:
            img = imread(img_path, as_gray=True)  # shape: (H, W)
            img = torch.from_numpy(img).float().unsqueeze(0) / 255.0  # (1, H, W)
        else:
            img = imread(img_path)  # shape: (H, W, C)
            if img.shape[-1] == 4:
                img = rgba2rgb_safe(img)
            img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # (C, H, W)

        mask = get_mask(self.mask_dir, idx)
        segmented_coral = Image.open(img_path).convert("L")  # or "RGB" if needed
        background = np.array(segmented_coral)

# only modify annotated pixels, keep coral structure for others
        mask[background == 0] = 255

        # ic("img stats:", img.min().item(), img.max().item(), img.mean().item())
        # ic("img shape: ", img.shape)
        # ic("mask shape: ", mask.shape)

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
                mask = mask.unsqueeze(0)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST).squeeze(0)

        return img, mask

    def __len__(self):
        return len(os.listdir(f"{self.img_dir}"))

# Create a dataset instance and data loader
dataset = CoralDataset('data/images-flouro', 'data/masks-flouro')
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
