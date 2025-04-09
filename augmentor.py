import os
import torch
import random
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image
from tqdm import tqdm
import numpy as np
from dataset.coral_dataset import get_coral_image
from utils.utils import generate_mask_and_label, rgba2rgb_safe

# Settings
img_dir = "data/images-flouro"
mask_dir = "data/masks-flouro"
out_img_dir = "data/out_images-flouro"
out_mask_dir = "data/out_masks-flouro"
N_AUGS = 5  # How many augmentations per image?

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)

def apply_augment(img, mask):
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
        mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
        mask = mask.squeeze(0)
    return img, mask

for idx in tqdm(range(1, len(os.listdir(img_dir)) + 1)):
    base_img = get_coral_image(img_dir, idx)
    base_mask = generate_mask_and_label(mask_dir, idx)
    # Save original (unaugmented) image + mask
    img_pil = TF.to_pil_image(base_img)
    mask_pil = Image.fromarray(base_mask.astype(np.uint8))

    save_name = f"{idx:04d}_orig.png"
    img_pil.save(os.path.join(out_img_dir, save_name))
    mask_pil.save(os.path.join(out_mask_dir, save_name))

    if base_img.shape[-1] == 4:
        base_img = rgba2rgb_safe(base_img)

    base_img = torch.from_numpy(base_img).float() / 255.0
    base_img = base_img.permute(2, 0, 1)
    base_mask = torch.from_numpy(base_mask).long()
    if base_mask.ndim == 3:
        base_mask = base_mask.squeeze(-1)


    save_name = f"{idx}.png"
    img_pil.save(os.path.join(out_img_dir, save_name))
    mask_pil.save(os.path.join(out_mask_dir, save_name))

