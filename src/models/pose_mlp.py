import torch
from torch import nn


class PoseMLP(nn.Module):
    """
    Order-invariant human action classifier.

    Input:
        (batch_size, num_frames, 99)

    Output:
        (batch_size, num_classes)
    """

    def __init__(
        self,
        input_size: int = 99,
        hidden_size: int = 128,
        num_classes: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "Expected input shape (batch, frames, features)."
            )

        # Average pose coordinates across the temporal dimension.
        pooled = x.mean(dim=1)

        return self.classifier(pooled)