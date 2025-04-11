import os
import torch
import cv2
from icecream import ic
import numpy as np
from skimage import io, filters, color, img_as_ubyte 
from skimage.io import imread

def rgba2rgb_safe(img):
    rgb = img[..., :3].astype(float)
    alpha = img[..., 3:] / 255.0
    white = np.ones_like(rgb) * 255
    out = rgb * alpha + white * (1 - alpha)
    return out.astype(np.uint8)


def get_coral_image(mask_dir, idx):
    # assert (mask_dir == "data/aug_images-flouro" or mask_dir == "data/images-non-flouro")
    
    files = sorted(os.listdir(mask_dir))
    img_path = os.path.join(mask_dir, files[idx])
    img = imread(img_path)

    return img

def remove_small_regions(mask, min_area=50, ignore_index=255):
    cleaned_mask = np.full_like(mask, ignore_index)

    for cls in np.unique(mask):
        if cls == ignore_index:
            continue

        class_mask = (mask == cls).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)

        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned_mask[labels == i] = cls

    return cleaned_mask

def find_color(img, color, tol=90):
    return np.all(np.abs(img[:, :, :3] - np.array(color)) <= tol, axis=2)


def get_size(start_path):
    print(start_path)
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            print(f)
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += 1

    return total_size
def generate_mask_and_label(mask_dir, idx):
    '''
    Takes a painted image and returns a mask with integer labels:
    - 0 = red (dead)
    - 1 = yellow (healing)
    - 2 = blue (bleached)
    Ignores near-black pixels.
    '''
    files = sorted(os.listdir(mask_dir))
    mask_path = os.path.join(mask_dir, files[idx])
    masked = imread(mask_path)

    if masked.shape[-1] == 4:
        masked = rgba2rgb_safe(mask_np)

    mask = torch.from_numpy(masked).long()

    if mask.ndim == 3:
        mask = mask.squeeze(-1)  # Remove channel dimension if present
    return mask


def get_class_frequencies(mask_dir, num_classes=3, ignore_index=255):
    class_counts = np.zeros(num_classes, dtype=np.int64)

    for file in sorted(os.listdir(mask_dir)):
        if not file.endswith(('.png', '.jpg')):
            continue
        mask = imread(os.path.join(mask_dir, file))

        if mask.ndim == 3:
            mask = mask[:, :, 0]  # if RGB mask, take one channel

        for cls in range(num_classes):
            class_counts[cls] += np.sum(mask == cls)

    return class_counts

def compute_class_weights(class_counts, epsilon=1e-6):
    total = class_counts.sum()
    weights = total / (class_counts + epsilon)  # prevent div by 0
    weights = weights / weights.sum()           # normalize to sum=1 (optional)
    return torch.tensor(weights, dtype=torch.float32)

