import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Down path
        x1 = self.down1(x)     # -> 64
        x2 = self.pool1(x1)

        x3 = self.down2(x2)    # -> 128
        x4 = self.pool2(x3)

        x5 = self.down3(x4)
        x6 = self.pool3(x5)

        x7 = self.bottleneck(x6)  # -> 256

        x8 = self.up3(x7)
        x8 = torch.cat([x8, x5], dim=1)
        x9 = self.upconv3(x8)

        x10 = self.up2(x9)
        x10 = torch.cat([x10, x3], dim=1)
        x11 = self.upconv2(x10)

        x12 = self.up1(x11)
        x12 = torch.cat([x12, x1], dim=1)
        x13 = self.upconv1(x12)

        return self.out(x13)

