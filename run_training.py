import os
from icecream import ic
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from train.train_model import train_model
from visualize_predictions import visualize_prediction
from models.segmentation_model import UNet
from utils.utils import *
from dataset.coral_dataset import CoralDataset
class ModelHyperparams:
    def __init__(self):
        self.lr = 1e-3
        self.batch_size = 10
        self.num_epochs = 15
        self.ignore_index = 255
        self.class_weights = torch.tensor([1.0, 1.5, 2.0])
        self.weight_decay = 1e-5

def main():
    cfg = ModelHyperparams()

    print("HELLOOO")
    dataset = CoralDataset(
    img_dir="data/aug_images-flouro",
    mask_dir="data/aug_masks-flouro",
    augment=True,
    
    )

    dataloader = DataLoader(dataset, cfg.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    # ic(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    class_counts = get_class_frequencies(dataset.mask_dir, num_classes=3)
    cfg.class_weights= compute_class_weights(class_counts)
     
    print(cfg.class_weights)

    criterion = nn.CrossEntropyLoss(ignore_index=255, weight=cfg.class_weights.to(device))
    train_model(model, dataloader, optimizer, criterion)
    for i in range(10):
        num = random.randint(1, len(dataset))
        visualize_prediction(model, dataset, device, num)

main()
