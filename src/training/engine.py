from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def run_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 1,
    description: str | None = None,
    input_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, float]:
    """
    Run one training or validation epoch.

    If optimizer is provided, training is performed.
    Otherwise, the model is evaluated without gradient updates.
    """

    is_training = optimizer is not None

    model.train(mode=is_training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    phase = "Train" if is_training else "Validation"

    progress = tqdm(
        dataloader,
        desc=description or f"{phase} | Epoch {epoch}",
        leave=False,
        dynamic_ncols=True,
    )

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for inputs, targets in progress:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if input_transform is not None:
                inputs = input_transform(inputs)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = targets.size(0)

            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            total_correct += (
                predictions == targets
            ).sum().item()

            total_samples += batch_size

            progress.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                acc=f"{total_correct / total_samples:.4f}",
            )

    if total_samples == 0:
        raise ValueError(
            f"{phase} dataloader is empty."
        )

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }