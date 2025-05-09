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
from visualize_predictions import save_results, visualize_prediction
from torch.utils.data import random_split
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
    train_dataset = CoralDataset(
        img_dir=config["img_dir"],
        mask_dir=config["mask_dir"],
        augment=config["augment"],
        grayscale=config["grayscale"],

    )

    in_channels = 1 if train_dataset.grayscale is True else 3

    total_size = len(train_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    if config["calculate_weights"] == True:

        class_counts = get_class_frequencies(train_dataset.mask_dir, num_classes=3)
        hyprparams.class_weights = compute_class_weights(class_counts)

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, hyprparams.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, hyprparams.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(in_channels)

    model = UNet(in_channels=in_channels).to(device)
    # ic(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hyprparams.lr, weight_decay=hyprparams.weight_decay)


    print(hyprparams.class_weights)

    criterion = nn.CrossEntropyLoss(
        ignore_index=255, weight=hyprparams.class_weights.to(device) if config["calculate_weights"] == True else None)

    avg_accuracy = 0
    num_visualizations = len(train_dataset) - 1
    train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, optimizer, criterion, config)
    save_results(model, train_dataset, val_dataset, device, train_losses, val_losses)
    for i in range(num_visualizations):
        num = i
        avg_accuracy += visualize_prediction(model, train_dataset, device, num, train_losses, val_losses)
    print(f"Average Accuracy: {avg_accuracy/num_visualizations}")


main()
