from __future__ import annotations

import torch
from torch import nn


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        from torchvision.models import resnet50

        self.network = resnet50(weights=None)
        original = self.network.conv1
        self.network.conv1 = nn.Conv2d(
            1,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=False,
        )
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)


class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        from torchvision.models import efficientnet_b0

        self.network = efficientnet_b0(weights=None)
        first = self.network.features[0][0]
        self.network.features[0][0] = nn.Conv2d(
            1,
            first.out_channels,
            kernel_size=first.kernel_size,
            stride=first.stride,
            padding=first.padding,
            bias=False,
        )
        classifier = self.network.classifier[-1]
        self.network.classifier[-1] = nn.Linear(
            classifier.in_features, num_classes
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)


class BRISCClassifier(nn.Module):
    def __init__(self, width: int = 32, dropout: float = 0.25, num_classes: int = 4) -> None:
        super().__init__()
        channels = (width, width * 2, width * 4, width * 8)
        layers: list[nn.Module] = []
        input_channels = 1
        for output_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(min(8, output_channels), output_channels),
                    nn.SiLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            input_channels = output_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(image))
