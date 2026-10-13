from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.settings import ACTION_CLASSES


def plot_training_history(
    history_path: Path,
    output_dir: Path,
) -> None:
    """
    Plot training and validation learning curves.
    """

    history_path = Path(history_path)
    output_dir = Path(output_dir)

    if not history_path.exists():
        raise FileNotFoundError(
            f"Training history not found: {history_path}"
        )

    history = pd.read_csv(history_path)

    required_columns = {
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
    }

    missing_columns = required_columns - set(history.columns)

    if missing_columns:
        raise ValueError(
            f"Missing history columns: {sorted(missing_columns)}"
        )

    if history.empty:
        raise ValueError("Training history is empty.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Loss curve
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        history["epoch"],
        history["train_loss"],
        label="Training Loss",
        linewidth=2,
    )

    ax.plot(
        history["epoch"],
        history["validation_loss"],
        label="Validation Loss",
        linewidth=2,
    )

    ax.set_title("CNN + LSTM — Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_dir / "loss_curve.png",
        dpi=200,
    )

    plt.close(fig)

    # Accuracy curve
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        history["epoch"],
        history["train_accuracy"] * 100,
        label="Training Accuracy",
        linewidth=2,
    )

    ax.plot(
        history["epoch"],
        history["validation_accuracy"] * 100,
        label="Validation Accuracy",
        linewidth=2,
    )

    ax.set_title("CNN + LSTM — Training and Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_dir / "accuracy_curve.png",
        dpi=200,
    )

    plt.close(fig)


def plot_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
    class_names: list[str] | tuple[str, ...] = ACTION_CLASSES,
) -> None:
    """
    Plot a confusion matrix with class labels.
    """

    matrix = np.asarray(matrix)
    output_path = Path(output_path)

    expected_shape = (len(class_names), len(class_names))

    if matrix.shape != expected_shape:
        raise ValueError(
            f"Expected confusion matrix shape {expected_shape}, "
            f"got {matrix.shape}."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )

    fig.colorbar(image, ax=ax)

    positions = np.arange(len(class_names))

    ax.set_xticks(positions)
    ax.set_yticks(positions)

    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("CNN + LSTM — Confusion Matrix")

    threshold = matrix.max() / 2 if matrix.size else 0

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]

            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)