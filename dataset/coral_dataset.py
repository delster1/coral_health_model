import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, color
from skimage.segmentation import watershed
from utils.utils import generate_mask_and_label, get_coral_image

class CoralDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __getitem__(self, idx):
        img = get_coral_image(self.img_dir, idx)
        if(img.shape[-1] == 4):
            img = color.rgba2rgb(img)
        mask, label = generate_mask_and_label(self.mask_dir, idx)
         # Create a target mask filled with 255 (ignore)
        target_mask = np.full(mask.shape, 255, dtype=np.uint8)
        target_mask[mask == 1] = label                 # Apply label only in region

        # Convert to tensors
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        target_mask = torch.tensor(target_mask, dtype=torch.long)
        return img, target_mask 

    def __len__(self):
        return len(os.listdir(f"{self.img_dir}"))

# Create a dataset instance and data loader
dataset = CoralDataset('data/images-flouro', 'data/masks-flouro')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

