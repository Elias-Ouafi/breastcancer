"""A compact 2D U-Net for single-channel MRI slice segmentation.

Standard encoder/decoder with skip connections and four downsampling steps, so the
input height and width must be divisible by 16 (the dataset resizes slices to a
fixed square size, e.g. 256, which satisfies this). The head returns raw logits;
apply ``torch.sigmoid`` for probabilities.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(conv -> GroupNorm -> ReLU) x2, the basic U-Net block.

    GroupNorm (not BatchNorm) is used deliberately. The lesion foreground is a tiny
    fraction of each slice and the effective batch is small, so BatchNorm's running
    mean/variance never converge to statistics that match inference: the saved model
    then collapses to sigmoid~0.5 everywhere in ``eval()`` mode (train-mode batch
    stats hid this, giving reproducible Dice only while ``model.train()`` was on).
    GroupNorm normalises per-sample over channel groups, so it behaves identically in
    train and eval mode and removes that instability entirely.
    """

    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        # Guard the group count so it always divides the channel width.
        groups = min(num_groups, out_ch)
        while out_ch % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)  # logits, shape (N, out_channels, H, W)
