import os
from icecream import ic
import numpy as np
from skimage import io, filters

def get_coral_image(mask_dir, idx):
    assert (mask_dir == "data/images-flouro" or mask_dir == "data/images-non-flouro")
    
    img = io.imread(f"{mask_dir}/{idx}.png") 

    return img

def generate_mask_and_label(mask_dir, idx):
    '''
    Takes a painted image (drawn over with rgb to identify regions)
    Returns the masked version and the label classification 
    (red - dead, green - healthy, blue - bleached)
    '''
    masked = io.imread(f"{mask_dir}/{idx}.png")
    red = (
        (masked[:, :, 0] == 255) &   # R
        (masked[:, :, 1] == 0) &   # G
        (masked[:, :, 2] == 0)   # B
        # Alpha is ignored (don't care)
    )
    green = (
        (masked[:, :, 0] == 0) &   # r
        (masked[:, :, 1] == 255) &   # g
        (masked[:, :, 2] == 0)   # b
        # alpha is ignored (don't care)
    )

    blue = (
        (masked[:, :, 0] == 0) &   # r
        (masked[:, :, 1] == 0) &   # g
        (masked[:, :, 2] == 255)   # b
        # alpha is ignored (don't care)
    )
    
    label = 0
    if (True in red):
        label = 1
        masked[~red] = [0,0,0,0]
    elif True in green:
        label = 2
        masked[~green] = [0,0,0,0]
    elif True in blue:
        label = 3
        masked[~blue] =  [0,0,0,0]
    else:
        raise(AssertionError("NO TRUE COLORS IN PAINTED IMAGE"))
    return masked, label

