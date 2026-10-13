from __future__ import annotations

import torch
from torch import nn

from src.config.settings import NUM_CLASSES


class CNNFeatureExtractor(nn.Module):
    """Extract spatial features from individual video frames."""

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class CNNLSTM(nn.Module):
    """CNN-LSTM architecture for video action recognition."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.cnn = CNNFeatureExtractor(
            feature_dim=feature_dim,
        )

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels, height, width = x.shape

        # Combine batch and time dimensions for CNN processing.
        x = x.reshape(
            batch_size * sequence_length,
            channels,
            height,
            width,
        )

        # Extract spatial features.
        features = self.cnn(x)

        # Restore temporal dimension.
        features = features.reshape(
            batch_size,
            sequence_length,
            -1,
        )

        # Learn temporal relationships.
        _, (hidden_state, _) = self.lstm(features)

        # Use the final LSTM hidden state.
        temporal_features = hidden_state[-1]

        # Produce classification logits.
        logits = self.classifier(temporal_features)

        return logits