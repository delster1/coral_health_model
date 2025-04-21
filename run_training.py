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
    def __init__(self, config):
        self.lr = float(config["lr"])
        self.batch_size = int(config["batch_size"])
        self.num_epochs = int(config["num_epochs"])
        self.ignore_index = 255
        self.class_weights = torch.tensor([1.0, 1.5, 2.0])
        self.weight_decay = float(config["weight_decay"])


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

    config = load_config(args.config)

    first_word = "grayscale-" if config["grayscale"] is True else "full-color-"
    second_word = os.path.basename(config["img_dir"])
    out_dir = first_word + second_word + '-' + config["checkpoint"]
    config["checkpoint_path"] = os.path.join(config["checkpoint_dir"], out_dir)
    
    hyprparams = ModelHyperparams(config)

    ic(f"Running Model with config:\n{config}")
    dataset = CoralDataset(
        img_dir=config["img_dir"],
        mask_dir=config["mask_dir"],
        augment=config["augment"],
        grayscale=config["grayscale"],

    )

    dataloader = DataLoader(dataset, hyprparams.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 1 if dataset.grayscale is True else 3
    print(in_channels)

    model = UNet(in_channels=in_channels).to(device)
    # ic(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hyprparams.lr, weight_decay=hyprparams.weight_decay)

    if config["calculate_weights"] == True:

        class_counts = get_class_frequencies(dataset.mask_dir, num_classes=3)
        hyprparams.class_weights = compute_class_weights(class_counts)

    print(hyprparams.class_weights)

    criterion = nn.CrossEntropyLoss(
        ignore_index=255, weight=hyprparams.class_weights.to(device) if config["calculate_weights"] == True else None)

    train_model(model, dataloader, optimizer, criterion, config)
    for i in range(10):
        num = random.randint(1, len(dataset) - 1)
        visualize_prediction(model, dataset, device, num)


main()
