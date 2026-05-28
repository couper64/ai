"""Convolutional neural network models for tensor classification."""

import torch
from torch import nn


class Classifier(nn.Module):
    """Classify image-like tensors with shape ``(batch, channels, height, width)``."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        # Backbone, i.e. feature extractor.
        # (N, 3, 224, 224)  →  conv  →  (N, 32, 224, 224)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # The classifier head is a Linear layer, which expects a 1D vector, not a 2D grid. You need to collapse the 28×28 spatial grid into one number per channel.
        # (N, 128, 28, 28)  →  pool  →  (N, 128, 1, 1).
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            # (N, 128, 1, 1)  →  flatten  →  (N, 128).
            nn.Flatten(),
            # Regulaisation.
            nn.Dropout(0.2),
            # Maps features → logits (one score per class).
            # (N, 128)  →  linear  →  (N, num_classes).
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of input tensors."""
        features = self.features(inputs)
        pooled = self.pool(features)
        return self.classifier(pooled)
