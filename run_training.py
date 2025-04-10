import os
from icecream import ic
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from train.train_model import train_model
from visualize_predictions import visualize_prediction
from models.segmentation_model import UNet
from utils.utils import *
from dataset.coral_dataset import CoralDataset
def main():
    print("HELLOOO")
    dataset = CoralDataset(
    img_dir="data/images-flouro",
    mask_dir="data/masks-flouro",
    
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    # ic(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    class_counts = get_class_frequencies(dataset.mask_dir, num_classes=3)
    class_weights = compute_class_weights(class_counts)
    print(class_weights)

    criterion = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights)

    train_model(model, dataloader, optimizer, criterion)
    for i in range(3):
        visualize_prediction(model, dataset, device, i)

main()
