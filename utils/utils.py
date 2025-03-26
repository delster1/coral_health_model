import os
from icecream import ic
import numpy as np
from skimage import io, filters, color

def get_coral_image(mask_dir, idx):
    assert (mask_dir == "data/images-flouro" or mask_dir == "data/images-non-flouro")
    
    img = io.imread(f"{mask_dir}/{idx}.png") 

    return img

def find_color(img, color, tol=60):
    return np.all(np.abs(img[:, :, :3] - np.array(color)) <= tol, axis=2)


def generate_mask_and_label(mask_dir, idx):
    '''
    Takes a painted image (drawn over with rgb to identify regions)
    Returns the masked version and the label classification 
    (red - dead, green - healthy, blue - bleached)
    '''
    masked = io.imread(f"{mask_dir}/{idx}.png")
    if(masked.shape[-1] == 4):
        masked = color.rgba2rgb(masked)
    target_mask = np.full(masked.shape[:2], 255, dtype=np.uint8)
    red = (
        (masked[:, :, 0] >= 225) &   # R
        (masked[:, :, 1] <= 50) &   # G
        (masked[:, :, 2] <= 50)   # B
        # Alpha is ignored (don't care)
    )
    yellow = (
        (masked[:, :, 0] >= 225) &   # r
        (masked[:, :, 1] >= 225) &   # g
        (masked[:, :, 2] <= 50)   # b
        # alpha is ignored (don't care)
    )

    blue = (
        (masked[:, :, 0] <= 50) &   # r
        (masked[:, :, 1] <= 50) &   # g
        (masked[:, :, 2] >= 225)   # b
        # alpha is ignored (don't care)
    )
    ic(red)
    ic(yellow)
    ic(blue)
    
    target_mask[~(red|yellow|blue)] = 255
    target_mask[red] = 0
    target_mask[yellow] = 1
    target_mask[blue] = 2

    ic(target_mask)
    assert (target_mask != 255).any(), "No labeled pixels found — target_mask is all 255!"
    num_labeled = np.sum(target_mask != 255)
    ic(f"{num_labeled} labeled pixels found.")


    return target_mask
