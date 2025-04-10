import os
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
    
    img = io.imread(f"{mask_dir}/{idx}.png") 

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
    masked = imread(f"{mask_dir}/{idx}.png")

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
