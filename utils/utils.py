import os
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

def find_color(img, color, tol=60):
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
    Takes a painted image (drawn over with rgb to identify regions)
    Returns the masked version and the label classification 
    (red - dead, green - healthy, blue - bleached)
    '''

    masked = imread(f"{mask_dir}/{idx}.png")

    if masked.shape[-1] == 4:
        masked = masked[:, :, :3]  # strip alpha

    # Initialize with ignore index (255)
    target_mask = np.full(masked.shape[:2], 255, dtype=np.uint8)


    red    = find_color(masked, (195, 75, 75))     # dead
    yellow = find_color(masked, (195, 195, 75))     # healing
    blue   = find_color(masked, (75, 75, 195))      # bleached

    # Assign labels only to known coral regions
    target_mask[red]    = 0
    target_mask[yellow] = 1
    target_mask[blue]   = 2

    assert (target_mask != 255).any(), "No labeled pixels found — target_mask is all 255!"
    return target_mask

