import os
import yaml
import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet on coral dataset")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def main():
    args = parse_args()
    print(args)

    hyprparams = ModelHyperparams()
    config = load_config(args.config)

    print("HELLOOO")
    dataset = CoralDataset(
        img_dir=config["img_dir"],
        mask_dir=config["mask_dir"],
        augment=config["augment"],
        grayscale=config["grayscale"],

    )

    dataloader = DataLoader(dataset, hyprparams.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 1 if dataset.grayscale else 3

    model = UNet(in_channels=in_channels).to(device)
    # ic(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hyprparams.lr, weight_decay=hyprparams.weight_decay)
    class_counts = get_class_frequencies(dataset.mask_dir, num_classes=3)
    hyprparams.class_weights = compute_class_weights(class_counts)

    print(hyprparams.class_weights)

    criterion = nn.CrossEntropyLoss(
        ignore_index=255, weight=hyprparams.class_weights.to(device))
    train_model(model, dataloader, optimizer, criterion, config)
    for i in range(10):
        num = random.randint(1, len(dataset))
        visualize_prediction(model, dataset, device, num)


main()
