from icecream import ic
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io, filters, color, img_as_ubyte
import random
from utils.utils import find_color

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

def test_generate_mask(mask_dir):

    ic(get_size(mask_dir))
    test_idx = random.randint(1,get_size(mask_dir))

    masked = io.imread(f"{mask_dir}/{test_idx}.png")
    if(masked.shape[-1] == 4):
        masked = color.rgba2rgb(masked)
        masked = img_as_ubyte(masked)
    non_black_pixels = np.where((masked[:, :, :3] != [0, 0, 0]).any(axis=2))
    ic(non_black_pixels)
    rgb_vals = masked[:, :, :3][(masked[:, :, :3] != [0, 0, 0]).any(axis=2)]
    ic(np.unique(rgb_vals, axis=0))

    target_mask = np.full(masked.shape[:2], 205, dtype=np.uint8)
    red    = find_color(masked, (195, 75, 75))    # Red = dead
    yellow = find_color(masked, (195, 195, 75))  # Yellow = healing
    blue   = find_color(masked, (75, 75, 195))    # Blue = bleached

    plt.imshow(red, cmap='Reds')
    plt.title("Red matched mask")
    plt.show()

    plt.imshow(yellow, cmap='Reds')
    plt.title("yell matched mask")
    plt.show()

    plt.imshow(blue, cmap='Reds')
    plt.title("blu matched mask")
    plt.show()

    ic(red)
    ic(yellow)
    ic(blue)
    ic(red.all() == yellow.all() and yellow.all() == blue.all())

    pixels = masked[:, :, :3].reshape(-1, 3)
    ic(np.unique(pixels, axis=0))

    assert (red != False).any(), "No labeled pixels found — target_mask is all 255!"

test_generate_mask("data/masks-flouro")

