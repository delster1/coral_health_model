import os
from collections import Counter
from icecream import ic
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def visualize_prediction(model, dataset, device, index): 
    model.eval()
    image, mask = dataset[index]
    with torch.no_grad():
        preds = model(image.unsqueeze(0).to(device)).squeeze(0).argmax(0).cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image.permute(1, 2, 0))
    ax[0].set_title("Image")
    ax[1].imshow(mask, cmap="tab10", vmin=0, vmax=3)
    ax[1].set_title("Ground Truth")
    ax[2].imshow(preds, cmap="tab10", vmin=0, vmax=2)
    print(f"Unique Prediction Values, {np.unique(preds)}")
    ax[2].set_title("Prediction")

    label_counts = Counter()

    for i in range(len(dataset)):
        _, mask = dataset[i]
        vals, counts = torch.unique(mask, return_counts=True)
        for v, c in zip(vals.tolist(), counts.tolist()):
            if v != 255:  # ignore mask
                label_counts[v] += c

    print(label_counts)
    plt.show()
