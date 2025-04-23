

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
from augmentor import *

mask_path = input("Enter image mask to convert:")
output_dir = input("Enter output directory")

mask_tensor = generate_mask_and_labelv2(mask_path)
mask_tensor = torch.from_numpy(mask_tensor)
save_mask(mask_tensor, mask_path, output_dir)


