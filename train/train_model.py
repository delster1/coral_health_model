from icecream import ic
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3  # e.g., bleached, healing, dead (exclude 255 for ignore)
ignore_index = 255

# Loss and optimizer


def train_model(model, dataloader, optimizer, criterion, config):
    '''
    Tensor Shape: 
        B - Batch size - # images at once
        C - Channels - number of feature maps/color channels to extract
        H - Height
        W - Width
    '''
    total_epochs = 0

    if (os.path.exists(config["checkpoint_path"])):
        checkpoint = torch.load(config["checkpoint_path"])
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Loaded from checkpoint Epoch # {checkpoint["epoch"]}")

        total_epochs = checkpoint["epoch"]
    outputs = None
    num_epochs = config["num_epochs"]
    for epoch in range(num_epochs):
        # ic(model.train())
        running_loss = 0.0

        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)                # Shape: (B, 3, H, W)
            masks = masks.to(device).long()           # Shape: (B, H, W)

            # Forward
            outputs = model(images)                   # Shape: (B, C, H, W)
            # with torch.no_grad():
            #     ic("output stats:", outputs.min().item(),
            #        outputs.max().item(), outputs.mean().item())

            # Loss
            masks = masks.squeeze(1).long()

            loss = criterion(outputs, masks)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        os.makedirs("checkpoints", exist_ok=True)
        save_dict = {'epoch': total_epochs + epoch,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss': loss, }
        print(f"Saved model to {config["checkpoint_path"]}, Total Epochs: {save_dict['epoch']}")
        torch.save(save_dict, config["checkpoint_path"])
