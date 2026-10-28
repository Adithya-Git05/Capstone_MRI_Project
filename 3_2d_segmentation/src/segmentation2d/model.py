from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = min(8, output_channels)
        while output_channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.layers(tensor)


class BRISCUNet(nn.Module):
    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.dec3 = ConvBlock(base_channels * 12, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 3, base_channels)
        self.output = nn.Conv2d(base_channels, 1, kernel_size=1)

    @staticmethod
    def _upsample(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(tensor, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        first = self.enc1(image)
        second = self.enc2(self.pool(first))
        third = self.enc3(self.pool(second))
        center = self.bottleneck(self.pool(third))
        decoded = self.dec3(torch.cat((third, self._upsample(center, third)), dim=1))
        decoded = self.dec2(torch.cat((second, self._upsample(decoded, second)), dim=1))
        decoded = self.dec1(torch.cat((first, self._upsample(decoded, first)), dim=1))
        return self.output(decoded)


class _DecoderBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = ConvBlock(input_channels + skip_channels, output_channels)

    def forward(
        self, tensor: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        tensor = F.interpolate(
            tensor, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.block(torch.cat((tensor, skip), dim=1))


class ResNet50UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import resnet50

        encoder = resnet50(weights=None)
        original = encoder.conv1
        encoder.conv1 = nn.Conv2d(
            1,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=False,
        )
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.dec3 = _DecoderBlock(2048, 1024, 256)
        self.dec2 = _DecoderBlock(256, 512, 128)
        self.dec1 = _DecoderBlock(128, 256, 64)
        self.dec0 = _DecoderBlock(64, 64, 32)
        self.head = nn.Sequential(
            ConvBlock(32, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        first = self.stem(image)
        second = self.layer1(self.pool(first))
        third = self.layer2(second)
        fourth = self.layer3(third)
        center = self.layer4(fourth)
        decoded = self.dec3(center, fourth)
        decoded = self.dec2(decoded, third)
        decoded = self.dec1(decoded, second)
        decoded = self.dec0(decoded, first)
        logits = self.head(decoded)
        return F.interpolate(
            logits, size=image.shape[-2:], mode="bilinear", align_corners=False
        )


class EfficientNetB0UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import efficientnet_b0

        encoder = efficientnet_b0(weights=None)
        first = encoder.features[0][0]
        encoder.features[0][0] = nn.Conv2d(
            1,
            first.out_channels,
            kernel_size=first.kernel_size,
            stride=first.stride,
            padding=first.padding,
            bias=False,
        )
        self.features = encoder.features
        self.dec4 = _DecoderBlock(1280, 320, 256)
        self.dec3 = _DecoderBlock(256, 112, 128)
        self.dec2 = _DecoderBlock(128, 40, 64)
        self.dec1 = _DecoderBlock(64, 24, 32)
        self.dec0 = _DecoderBlock(32, 16, 32)
        self.head = nn.Sequential(
            ConvBlock(32, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tensor = self.features[0](image)
        skip0 = self.features[1](tensor)
        skip1 = self.features[2](skip0)
        skip2 = self.features[3](skip1)
        tensor = self.features[4](skip2)
        skip3 = self.features[5](tensor)
        tensor = self.features[6](skip3)
        skip4 = self.features[7](tensor)
        center = self.features[8](skip4)
        decoded = self.dec4(center, skip4)
        decoded = self.dec3(decoded, skip3)
        decoded = self.dec2(decoded, skip2)
        decoded = self.dec1(decoded, skip1)
        decoded = self.dec0(decoded, skip0)
        logits = self.head(decoded)
        return F.interpolate(
            logits, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
