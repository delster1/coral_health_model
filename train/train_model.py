from icecream import ic
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3  # e.g., bleached, healing, dead (exclude 255 for ignore)
ignore_index = 255

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

def train_model(model, dataloader, optimizer, criterion, epochs=10):
    '''
    Tensor Shape: 
        B - Batch size - # images at once
        C - Channels - number of feature maps/color channels to extract
        H - Height
        W - Width
    '''
    num_epochs = 10
    for epoch in range(num_epochs):
        ic(model.train())
        running_loss = 0.0

        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)                # Shape: (B, 3, H, W)
            masks = masks.to(device).long()           # Shape: (B, H, W)

            # Forward
            outputs = model(images)                   # Shape: (B, C, H, W)

            # Loss
            loss = criterion(outputs, masks)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

