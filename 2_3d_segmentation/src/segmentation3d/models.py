from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F
from torch import nn


class _ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm3d(out_channels)
        self.skip = (nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm3d(out_channels),
        ))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.skip(value)
        value = self.activation(self.norm1(self.conv1(value)))
        return self.activation(self.norm2(self.conv2(value)) + residual)


class _MBConv3D(nn.Module):
    """Compact mobile inverted bottleneck with depthwise 3D convolution."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        expanded = in_channels * 4
        squeeze = max(8, expanded // 4)
        self.expand = nn.Sequential(nn.Conv3d(in_channels, expanded, 1, bias=False),
                                    nn.BatchNorm3d(expanded), nn.SiLU(inplace=True))
        self.depthwise = nn.Sequential(
            nn.Conv3d(expanded, expanded, 3, stride, 1, groups=expanded, bias=False),
            nn.BatchNorm3d(expanded), nn.SiLU(inplace=True))
        self.squeeze_excite = nn.Sequential(nn.AdaptiveAvgPool3d(1),
                                            nn.Conv3d(expanded, squeeze, 1),
                                            nn.SiLU(inplace=True),
                                            nn.Conv3d(squeeze, expanded, 1), nn.Sigmoid())
        self.project = nn.Sequential(nn.Conv3d(expanded, out_channels, 1, bias=False),
                                     nn.BatchNorm3d(out_channels))
        self.skip = (nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm3d(out_channels),
        ))
        self.activation = nn.SiLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.skip(value)
        value = self.depthwise(self.expand(value))
        return self.activation(self.project(value * self.squeeze_excite(value)) + residual)


class _DenseSegmentation3D(nn.Module):
    """Small 3D encoder/decoder baseline with explicit skip connections."""

    def __init__(self, in_channels: int, out_channels: int, feature_size: int,
                 block: type[nn.Module]) -> None:
        super().__init__()
        widths = [feature_size * scale for scale in (1, 2, 4, 8, 16)]
        self.encoder = nn.ModuleList([block(in_channels, widths[0])] + [
            block(widths[index - 1], widths[index], stride=2) for index in range(1, 5)
        ])
        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for index in range(4, 0, -1):
            self.up.append(nn.ConvTranspose3d(widths[index], widths[index - 1], 2, 2))
            self.decoder.append(block(widths[index - 1] * 2, widths[index - 1]))
        self.head = nn.Conv3d(widths[0], out_channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        skips = []
        for stage in self.encoder:
            value = stage(value)
            skips.append(value)
        value = skips.pop()
        for up, stage in zip(self.up, self.decoder):
            value = up(value)
            skip = skips.pop()
            if value.shape[2:] != skip.shape[2:]:
                value = F.interpolate(value, size=skip.shape[2:], mode="trilinear", align_corners=False)
            value = stage(torch.cat((value, skip), dim=1))
        return self.head(value)


def build_monai_model(
    model_name: str,
    *,
    patch_size: Sequence[int],
    in_channels: int,
    out_channels: int,
    feature_size: int = 24,
) -> nn.Module:
    if model_name == "segresnet":
        from monai.networks.nets import SegResNet

        return SegResNet(
            spatial_dims=3,
            init_filters=feature_size,
            in_channels=in_channels,
            out_channels=out_channels,
        )
    if model_name == "swinunetr":
        from monai.networks.nets import SwinUNETR

        return SwinUNETR(
            img_size=tuple(int(value) for value in patch_size),
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=True,
        )
    if model_name == "resnet3d":
        return _DenseSegmentation3D(in_channels, out_channels, feature_size, _ResidualBlock3D)
    if model_name == "efficientnet3d":
        return _DenseSegmentation3D(in_channels, out_channels, feature_size, _MBConv3D)
    raise ValueError(f"Unsupported 3D segmentation model: {model_name}")
