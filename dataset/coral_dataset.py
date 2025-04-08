import os
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
from utils.utils import generate_mask_and_label, get_coral_image, rgba2rgb_safe

class CoralDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __getitem__(self, idx):
        idx += 1
        img = get_coral_image(self.img_dir, idx)

        if(img.shape[-1] == 4):
            img = rgba2rgb_safe(img)


        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2,0,1)

        ic("img stats:", img.min().item(), img.max().item(), img.mean().item())


        mask = generate_mask_and_label(self.mask_dir, idx)
        mask = torch.from_numpy(mask).long()

        if mask.ndim == 3:
            mask = mask.squeeze(-1)  # Remove channel dimension if present

         # Create a target mask filled with 255 (ignore)

        ic(img.shape)
        # ic(img)
        # ic(mask.shape)
        # Convert to tensors
        return img, mask 

    def __len__(self):
        return len(os.listdir(f"{self.img_dir}"))

# Create a dataset instance and data loader
dataset = CoralDataset('data/images-flouro', 'data/masks-flouro')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

