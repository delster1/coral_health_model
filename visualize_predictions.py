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
import pandas as pd
import torch
from tqdm import tqdm

def save_results(model, training_dataset, val_dataset, device, train_losses, val_losses, save_path="results.xlsx"):
    model.eval()
    
    def collect_dataset_info(dataset, name):
        results = []
        for idx in tqdm(range(len(dataset)), desc=f"Processing {name}"):
            image, mask = dataset[idx]
            with torch.no_grad():
                pred = model(image.unsqueeze(0).to(device))
                pred = pred.squeeze(0).argmax(0).cpu()

            correct = (pred == mask).sum().item()
            total = mask.numel()
            accuracy = correct / total * 100

            results.append({
                "Dataset": name,
                "Index": idx,
                "Accuracy (%)": accuracy
            })
        return results

    # Collect per-sample accuracy stats
    train_results = collect_dataset_info(training_dataset, "Train")
    val_results = collect_dataset_info(val_dataset, "Validation")

    # Combine results
    full_results = train_results + val_results
    df = pd.DataFrame(full_results)

    # Add per-epoch losses as separate sheet or table
    losses_df = pd.DataFrame({
        "Epoch": list(range(1, len(train_losses)+1)),
        "Train Loss": train_losses,
        "Validation Loss": val_losses
    })

    # Save to Excel with multiple sheets
    with pd.ExcelWriter(save_path) as writer:
        df.to_excel(writer, index=False, sheet_name="Per Sample Accuracy")
        losses_df.to_excel(writer, index=False, sheet_name="Losses Over Time")

    print(f"[✓] Results saved to: {save_path}")


def visualize_prediction(model, dataset, device, index, train_losses, val_losses): 
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
    fig, ax = plt.subplots(1,5,figsize=(25,5)) 
    # Image
    ax[0].imshow(image.cpu().permute(1, 2, 0))
    ax[0].set_title("Image")
    ax[0].axis('off')

    # Ground Truth
    ax[1].imshow(mask, cmap="tab10", vmin=0, vmax=3)
    ax[1].set_title("Ground Truth")
    ax[1].axis('off')

    # Prediction
    ax[2].imshow(preds, cmap="tab10", vmin=0, vmax=3)
    ax[2].set_title(f"Prediction\nAccuracy: {accuracy_percent:.2f}%")
    ax[2].axis('off')

    # Training Loss
    ax[3].plot(train_losses, label='Training Loss', color='blue')
    ax[3].set_xlabel('Epoch')
    ax[3].set_ylabel('Loss')
    ax[3].set_title('Training Loss Over Time')
    ax[3].legend()

    # Validation Loss
    ax[4].plot(val_losses, label='Validation Loss', color='orange')
    ax[4].set_xlabel('Epoch')
    ax[4].set_ylabel('Loss')
    ax[4].set_title('Validation Loss Over Time')
    ax[4].legend()

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
    return accuracy_percent

