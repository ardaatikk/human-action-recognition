import numpy as np
import pandas as pd
import pytest

from src.training.plots import (
    plot_training_history,
    plot_confusion_matrix,
)


def create_sample_history(path):
    history = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_loss": [1.8, 1.4, 1.1],
            "train_accuracy": [0.2, 0.4, 0.6],
            "validation_loss": [1.9, 1.5, 1.2],
            "validation_accuracy": [0.2, 0.3, 0.5],
        }
    )

    history.to_csv(path, index=False)


def test_training_history_creates_figures(tmp_path):
    history_path = tmp_path / "history.csv"
    output_dir = tmp_path / "figures"

    create_sample_history(history_path)

    plot_training_history(
        history_path,
        output_dir,
    )

    assert (output_dir / "loss_curve.png").exists()
    assert (output_dir / "accuracy_curve.png").exists()


def test_training_history_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        plot_training_history(
            tmp_path / "missing.csv",
            tmp_path / "figures",
        )


def test_training_history_missing_columns(tmp_path):
    history_path = tmp_path / "history.csv"

    pd.DataFrame(
        {"epoch": [1, 2, 3]}
    ).to_csv(history_path, index=False)

    with pytest.raises(ValueError, match="Missing history columns"):
        plot_training_history(
            history_path,
            tmp_path / "figures",
        )


def test_training_history_empty_file(tmp_path):
    history_path = tmp_path / "history.csv"

    pd.DataFrame(
        columns=[
            "epoch",
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
        ]
    ).to_csv(history_path, index=False)

    with pytest.raises(ValueError, match="empty"):
        plot_training_history(
            history_path,
            tmp_path / "figures",
        )


def test_confusion_matrix_creates_figure(tmp_path):
    matrix = np.array(
        [
            [5, 1, 0],
            [0, 4, 2],
            [1, 0, 5],
        ]
    )

    output_path = tmp_path / "confusion_matrix.png"

    plot_confusion_matrix(
        matrix,
        output_path,
        class_names=["boxing", "walking", "running"],
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_confusion_matrix_invalid_shape(tmp_path):
    matrix = np.array(
        [
            [5, 1],
            [0, 4],
        ]
    )

    with pytest.raises(ValueError, match="Expected confusion matrix shape"):
        plot_confusion_matrix(
            matrix,
            tmp_path / "confusion_matrix.png",
            class_names=["boxing", "walking", "running"],
        )