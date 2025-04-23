import matplotlib.pyplot as plt
import torch
from utils.utils import get_mask, get_coral_image, rgba2rgb_safe, generate_mask_and_labelv2
from skimage.io import imread
from augmentor import *
import sys
import os

def show_mask(mask_tensor):

    plt.imshow(mask_tensor.numpy(), cmap='tab10', vmin=0, vmax=3)  # or 'nipy_spectral'
    plt.colorbar()
    plt.title("Mask Visualization")
    plt.show()

def get_mask_tensor(mask_path):

    if mask_path[-1] == "/":
        files = sorted(os.listdir(mask_path))  # Ensure matching order
        for idx, file in enumerate(files):
            mask_dir = os.path.join(mask_path, files[idx])
            mask_np = imread(mask_dir)

            if mask_np.shape[-1] == 4:
                mask_np = rgba2rgb_safe(mask_np)

            mask = torch.from_numpy(mask_np).long()

            if mask.ndim == 3:
                mask = mask.squeeze(-1)  # Remove channel dimension if present


            show_mask(mask)
        return

    mask_np = imread(mask_path)

    if mask_np.shape[-1] == 4:
        mask_np = rgba2rgb_safe(mask_np)

    mask = torch.from_numpy(mask_np).long()

    if mask.ndim == 3:
        mask = mask.squeeze(-1)  # Remove channel dimension if present

    show_mask(mask)

def create_masks(mask_path):
    if mask_path[-1] == "/":
        print("WHAST THE FUCK")
        files = sorted(os.listdir(mask_path))  # Ensure matching order
        for idx, file in enumerate(files):
            mask_dir = os.path.join(mask_path, files[idx])
            mask = generate_mask_and_labelv2(mask_dir)

            print(f"WHAST THE {file}")
            mask_pil = Image.fromarray(mask.astype(np.uint8))
            mask_pil.save(os.path.join(mask_path,files[idx] ))

print(sys.argv[1])
get_mask_tensor(sys.argv[1])

