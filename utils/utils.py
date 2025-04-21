import os
from PIL import Image
import torch
import cv2
from icecream import ic
import numpy as np
from skimage import io, filters, color, img_as_ubyte 
from skimage.io import imread

def clear_output_folder(out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        return
    for filename in os.listdir(out_dir):
        file_path = os.path.join(out_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
def save_prediction_mask(mask_tensor, save_path):
    mask_np = mask_tensor.astype(np.uint8)
    mask_img = Image.fromarray(mask_np)
    mask_img.save(save_path)

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
def get_mask(mask_dir, idx):
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
        masked = rgba2rgb_safe(masked)

    # Convert to torch tensor
    mask = torch.from_numpy(masked).long()

    # If RGB image, convert to grayscale/intensity map
    if mask.ndim == 3:
        # Identify black pixels: R=G=B=0
        black_pixels = (mask == 0).all(dim=-1)

        # Convert RGB to grayscale-like integer (just pick one channel here)
        mask = mask[..., 0]  # assuming R=G=B so any channel works

        # Set black pixels to 255
        mask[black_pixels] = 255

    # Set out-of-bound class IDs (>2) to 255 (ignore index)
    mask[(mask > 2) & (mask != 255)] = 255

    return mask
def generate_mask_and_labelv2(mask_path):
    '''
    Takes a painted image and returns a mask with integer labels:
    - 0 = red (dead)
    - 1 = yellow (healing)
    - 2 = blue (bleached)
    Ignores near-black pixels.
    '''
    masked = imread(mask_path)

    if masked.shape[-1] == 4:
        masked = masked[:, :, :3]  # strip alpha

    H, W, _ = masked.shape
    target_mask = np.full((H, W), 255, dtype=np.uint8)

    # Target RGB colors
    red    = np.array([195, 60, 60])
    yellow = np.array([195, 195, 60])
    blue   = np.array([60, 60, 195])

    # Reshape for vectorized distance computation
    flat_img = masked.reshape(-1, 3)

    # Compute L2 distance to each class color
    dist_red    = np.linalg.norm(flat_img - red, axis=1)
    dist_yellow = np.linalg.norm(flat_img - yellow, axis=1)
    dist_blue   = np.linalg.norm(flat_img - blue, axis=1)

    # Stack and get argmin for closest color
    distances = np.stack([dist_red, dist_yellow, dist_blue], axis=1)
    min_indices = np.argmin(distances, axis=1)

    # # Optional: only assign if pixel isn’t close to black
    brightness = np.linalg.norm(flat_img, axis=1)
    not_black = brightness > 0  # tweak threshold as needed

    # Fill mask
    target_mask_flat = target_mask.flatten()
    target_mask_flat[not_black] = min_indices[not_black]
    target_mask = target_mask_flat.reshape(H, W)

    assert (target_mask != 255).any(), "No labeled pixels found — mask may be too dark or empty!"
    target_mask = remove_small_regions(target_mask)
    return target_mask

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

