
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from dataset.coral_dataset import CoralDataset


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )

        self.up1 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear'), -- will need to be intentional about up & down sampling later
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_out = nn.Conv2d(64, 3, kernel_size=1)  # <-- 3 classes: dead, healthy, bleached

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = self.down1(x)
        x = self.up1(x)
        return self.conv_out(x)

