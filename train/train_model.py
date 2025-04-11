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
criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

def train_model(model, dataloader, optimizer, criterion):
    '''
    Tensor Shape: 
        B - Batch size - # images at once
        C - Channels - number of feature maps/color channels to extract
        H - Height
        W - Width
    '''
    outputs = None
    num_epochs = 90
    for epoch in range(num_epochs):
        # ic(model.train())
        running_loss = 0.0

        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)                # Shape: (B, 3, H, W)
            unique_vals = torch.unique(masks)
            print("Mask Values: ", unique_vals)
            masks = masks.to(device).long()           # Shape: (B, H, W)

            # Forward
            outputs = model(images)                   # Shape: (B, C, H, W)
            with torch.no_grad():
                ic("output stats:", outputs.min().item(), outputs.max().item(), outputs.mean().item())
            preds = torch.argmax(outputs, dim=1)

            plt.imshow(preds[0].cpu().numpy(), cmap='viridis')

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
        torch.save(model.state_dict(), "checkpoints/unet_epoch10.pth")

