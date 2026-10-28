from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.segmentation2d.model import ConvBlock


class AttentionGate(nn.Module):
    def __init__(self, skip_channels: int, gate_channels: int) -> None:
        super().__init__()
        hidden = max(1, min(skip_channels, gate_channels) // 2)
        self.skip_proj = nn.Conv2d(skip_channels, hidden, 1, bias=False)
        self.gate_proj = nn.Conv2d(gate_channels, hidden, 1, bias=False)
        self.score = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate = F.interpolate(
            gate,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        attention = self.score(self.skip_proj(skip) + self.gate_proj(gate))
        return skip * attention


class BRISCAttentionUNet(nn.Module):
    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        channels = base_channels

        self.enc1 = ConvBlock(1, channels)
        self.enc2 = ConvBlock(channels, channels * 2)
        self.enc3 = ConvBlock(channels * 2, channels * 4)
        self.bottleneck = ConvBlock(channels * 4, channels * 8)
        self.pool = nn.MaxPool2d(2)

        self.att3 = AttentionGate(channels * 4, channels * 8)
        self.att2 = AttentionGate(channels * 2, channels * 4)
        self.att1 = AttentionGate(channels, channels * 2)

        self.dec3 = ConvBlock(channels * 12, channels * 4)
        self.dec2 = ConvBlock(channels * 6, channels * 2)
        self.dec1 = ConvBlock(channels * 3, channels)
        self.output = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        first = self.enc1(image)
        second = self.enc2(self.pool(first))
        third = self.enc3(self.pool(second))
        center = self.bottleneck(self.pool(third))

        up3 = F.interpolate(
            center, size=third.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded = self.dec3(torch.cat((self.att3(third, up3), up3), dim=1))

        up2 = F.interpolate(
            decoded, size=second.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded = self.dec2(torch.cat((self.att2(second, up2), up2), dim=1))

        up1 = F.interpolate(
            decoded, size=first.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded = self.dec1(torch.cat((self.att1(first, up1), up1), dim=1))
        return self.output(decoded)
