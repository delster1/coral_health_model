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

AUGMENT = False
# ---------------- Settings ---------------- #
IMG_DIR = "new_data/images-flouro"
MASK_DIR = "new_data/masks-flouro"
OUT_IMG_DIR = "new_data/out_images-flouro"
OUT_MASK_DIR = "new_data/out_masks-flouro"
N_AUGS = 5  # How many augmentations per image?

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

# ---------------- Image Functions ---------------- #
def load_image(img_dir, idx):
    img = get_coral_image(img_dir, idx)
    if img.shape[-1] == 4:
        img = rgba2rgb_safe(img)
    img_tensor = torch.from_numpy(img).float() / 255.0
    return img_tensor.permute(2, 0, 1)  # HWC → CHW

def save_image(img_tensor, save_name, out_img_dir):
    img_pil = TF.to_pil_image(img_tensor)
    img_pil.save(os.path.join(out_img_dir, save_name))

def augment_image(img_tensor):
    img = img_tensor.clone()
    if random.random() > 0.5:
        img = TF.hflip(img)
    if random.random() > 0.5:
        img = TF.vflip(img)
    if random.random() > 0.5:
        angle = random.uniform(-30, 30)
        img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
    return img


# ---------------- Mask Functions ---------------- #
def load_mask(mask_dir, idx):
    mask_np = generate_mask_and_label(mask_dir, idx)
    mask_tensor = torch.from_numpy(mask_np).long()
    if mask_tensor.ndim == 3:
        mask_tensor = mask_tensor.squeeze(-1)
    return mask_tensor

def save_mask(mask_tensor, save_name, out_mask_dir):
    mask_pil = Image.fromarray(mask_tensor.numpy().astype(np.uint8))
    mask_pil.save(os.path.join(out_mask_dir, save_name))

def augment_mask(mask_tensor):
    mask = mask_tensor.clone()
    if random.random() > 0.5:
        mask = TF.hflip(mask)
    if random.random() > 0.5:
        mask = TF.vflip(mask)
    if random.random() > 0.5:
        angle = random.uniform(-30, 30)
        mask = mask.unsqueeze(0)
        mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
        mask = mask.squeeze(0)
    return mask


# ---------------- Workflow Functions ---------------- #
def process_single_image_mask_pair(idx, n_augs=N_AUGS):
    img = load_image(IMG_DIR, idx)
    mask = load_mask(MASK_DIR, idx)

    # Save original
    base_name = f"{idx:04d}_orig.png"
    save_image(img, base_name, OUT_IMG_DIR)
    save_mask(mask, base_name, OUT_MASK_DIR)
    if AUGMENT == True:
        for aug_idx in range(n_augs):
            img_aug = augment_image(img)
            mask_aug = augment_mask(mask)

            aug_name = f"{idx:04d}_aug{aug_idx+1}.png"
            save_image(img_aug, aug_name, OUT_IMG_DIR)
            save_mask(mask_aug, aug_name, OUT_MASK_DIR)
def run_augmentation_pipeline():
    n_images = len(os.listdir(IMG_DIR))
    for idx in tqdm(range(1, n_images + 1)):
        process_single_image_mask_pair(idx)


# ---------------- Entry Point ---------------- #
if __name__ == "__main__":
    run_augmentation_pipeline()

