import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.1, p=0.0):
        super(DoubleConv, self).__init__()
        if p <= 0.0 or p > 1.0:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 3, padding="same",
                    padding_mode="zeros"
                ),
                # nn.ReLU(inplace=True),
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                nn.Conv2d(
                    out_channels, out_channels, 3, padding="same",
                    padding_mode="zeros"
                ),
                # nn.ReLU(inplace=True)
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 3, padding="same",
                    padding_mode="zeros"
                ),
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                nn.Conv2d(
                    out_channels, out_channels, 3, padding="same",
                    padding_mode="zeros"
                ),
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                nn.Dropout(p=p, inplace=False),
            )

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=3,
        features=[64, 128, 256, 512, 1024],
        ps=[0.0, 0.0, 0.0, 0.2, 0.2],
    ):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        for (feature, p) in zip(features, ps):
            self.downs.append(DoubleConv(in_channels, feature, p=p))
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features[:-1]):
            self.ups.append(Up(in_channels, feature))
            in_channels = feature

        # Final convolution layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Down part of UNet
        for down in self.downs[:-1]:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.downs[-1](x)
        skip_connections = skip_connections[::-1]

        # Up part of UNet
        for skip_connection, up in zip(skip_connections, self.ups):
            x = up(x, skip_connection)

        # Final convolution layer
        x = self.final_conv(x)

        return x
