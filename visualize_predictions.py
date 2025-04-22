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
from utils.utils import save_prediction_mask, clear_output_folder
def visualize_prediction(model, dataset, device, index): 
    model.eval()
    image, mask = dataset[index]
    
    image = image.to(device)

    with torch.no_grad():
        output = model(image.unsqueeze(0))  # Add batch dimension
        preds = output.squeeze(0).argmax(0).cpu().numpy()  # Predicted classes per pixel

    # save predicted mask
    clear_output_folder("outputs")
    save_prediction_mask(preds, "outputs/pred_img.png")

    # modify where annotated
    preds_masked = preds.copy()
    preds_masked[mask == 255] = 255

    # pixel-wise accuracy
    condition = (mask != 255)
    correct = (preds_masked == mask) & condition
    total = condition.sum().item()
    num_correct = correct.sum().item()

    accuracy_percent = 100.0 * num_correct / total if total > 0 else 0.0

    print(f"Unique Prediction Values: {np.unique(preds)}")

    # plotting
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image.cpu().permute(1, 2, 0))
    ax[0].set_title("Image")
    ax[0].axis('off')

    ax[1].imshow(mask, cmap="tab10", vmin=0, vmax=3)
    ax[1].set_title("Ground Truth")
    ax[1].axis('off')

    ax[2].imshow(preds, cmap="tab10", vmin=0, vmax=3)
    ax[2].set_title(f"Prediction\nAccuracy: {accuracy_percent:.2f}%")
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()

    label_counts = Counter()
    for i in range(len(dataset)):
        _, mask_i = dataset[i]
        vals, counts = torch.unique(mask_i, return_counts=True)
        for v, c in zip(vals.tolist(), counts.tolist()):
            if v != 255:
                label_counts[v] += c

    print(f"Label counts across dataset: {label_counts}")

