from __future__ import annotations

import torch
from torch import nn

from src.config.settings import NUM_CLASSES
from src.features.pose_normalization import NUM_POSE_FEATURES


class PoseLSTM(nn.Module):
    """
    LSTM-based human action recognition using pose landmarks.

    Input:
        (batch_size, sequence_length, 135)

    Output:
        (batch_size, num_classes)
    """

    def __init__(
        self,
        input_size: int = NUM_POSE_FEATURES,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be positive")

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input (B, T, F), got {tuple(x.shape)}"
            )

        if x.shape[1] == 0:
            raise ValueError("Sequence length must be positive")

        if x.shape[2] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, "
                f"got {x.shape[2]}"
            )

        lstm_output, (hidden_state, cell_state) = self.lstm(x)

        final_hidden_state = hidden_state[-1]

        final_hidden_state = self.dropout(
            final_hidden_state
        )

        logits = self.classifier(final_hidden_state)

        return logits