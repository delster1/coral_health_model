import os
from icecream import ic
import numpy as np
from skimage import io, filters, color

def get_coral_image(mask_dir, idx):
    assert (mask_dir == "data/images-flouro" or mask_dir == "data/images-non-flouro")
    
    img = io.imread(f"{mask_dir}/{idx}.png") 

    return img

def find_color(img, rgb, tol=10):
    return np.all(np.abs(img[:, :, :3] - np.array(rgb)) <= tol, axis=-1)


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
    red = find_color(masked, (255, 0,0))
    yellow = find_color(masked, (255, 255,0))
    blue = find_color(masked, (0, 0,255))
    
    target_mask[red] = 0
    target_mask[yellow] = 1
    target_mask[blue] = 2
    target_mask[~(red|yellow|blue)] = 255

    return target_mask
