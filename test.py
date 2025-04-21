from icecream import ic
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io, filters, color, img_as_ubyte
import random
from utils.utils import *


print("WHAT THE FUCK")
mask = generate_mask_and_labelv2("new_masks/19.png")
print("READ MASK")
plt.imshow(mask, cmap="tab10", vmin=0, vmax=3)
plt.show()

# img = Image.open("new_masks/19.png").convert("RGBA")
# rgba = np.array(img)
#
# # Extract alpha channel
# alpha = rgba[:, :, 3]
#
# # Show alpha as grayscale
# plt.imshow(alpha, cmap="gray")
# plt.title("Alpha Channel (Transparency)")
# plt.axis("off")
# plt.show()
