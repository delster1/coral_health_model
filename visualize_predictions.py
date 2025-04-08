import os
from icecream import ic
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def visualize_prediction(model, dataset, device, index, outputs):
    model.eval()
    image, mask = dataset[index]
    with torch.no_grad():
        preds = torch.argmax(outputs, dim=1)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image.permute(1, 2, 0))
    ax[0].set_title("Image")
    ax[1].imshow(mask)
    ax[1].set_title("Ground Truth")
    ax[2].imshow(pred)
    ax[2].set_title("Prediction")
    plt.show()

