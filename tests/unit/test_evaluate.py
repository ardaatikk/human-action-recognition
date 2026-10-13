from __future__ import annotations

import csv
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config.settings import ACTION_CLASSES, NUM_CLASSES
from src.training.evaluate import evaluate_model, save_results
from src.training.metrics import calculate_classification_metrics
from src.training.train import build_dataset, build_model


class DummyClassifier(torch.nn.Module):
    """A deterministic classifier for evaluation tests."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


def test_evaluate_model_predictions_and_probabilities():
    logits = torch.tensor(
        [
            [4.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    labels = torch.tensor([0, 1, 2])

    dataloader = DataLoader(
        TensorDataset(logits, labels),
        batch_size=2,
        shuffle=False,
    )

    y_true, y_pred, probabilities = evaluate_model(
        model=DummyClassifier(),
        dataloader=dataloader,
        device=torch.device("cpu"),
    )

    assert y_true.shape == (3,)
    assert y_pred.shape == (3,)
    assert probabilities.shape == (3, NUM_CLASSES)

    np.testing.assert_array_equal(y_true, [0, 1, 2])
    np.testing.assert_array_equal(y_pred, [0, 1, 2])

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(3),
        atol=1e-6,
    )


def test_evaluate_model_handles_incorrect_predictions():
    logits = torch.tensor(
        [
            [0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    labels = torch.tensor([0, 2])

    dataloader = DataLoader(
        TensorDataset(logits, labels),
        batch_size=2,
    )

    y_true, y_pred, _ = evaluate_model(
        model=DummyClassifier(),
        dataloader=dataloader,
        device=torch.device("cpu"),
    )

    metrics = calculate_classification_metrics(
        y_true,
        y_pred,
        ACTION_CLASSES,
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["confusion_matrix"][0, 1] == 1
    assert metrics["confusion_matrix"][2, 2] == 1


def test_save_results_creates_expected_files(tmp_path):
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 1])

    probabilities = np.array(
        [
            [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.8, 0.1, 0.0, 0.0, 0.0],
            [0.1, 0.7, 0.2, 0.0, 0.0, 0.0],
        ]
    )

    metrics = calculate_classification_metrics(
        y_true,
        y_pred,
        ACTION_CLASSES,
    )

    save_results(
        output_dir=tmp_path,
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
    )

    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "confusion_matrix.npy").exists()
    assert (tmp_path / "predictions.csv").exists()

    with (tmp_path / "metrics.json").open(
        encoding="utf-8"
    ) as file:
        saved_metrics = json.load(file)

    assert saved_metrics["accuracy"] == metrics["accuracy"]

    saved_matrix = np.load(
        tmp_path / "confusion_matrix.npy"
    )

    np.testing.assert_array_equal(
        saved_matrix,
        metrics["confusion_matrix"],
    )

    with (tmp_path / "predictions.csv").open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 3
    assert rows[0]["true_class"] == ACTION_CLASSES[0]
    assert rows[2]["predicted_class"] == ACTION_CLASSES[1]


def test_build_model_supports_both_architectures():
    from src.models.cnn_lstm import CNNLSTM
    from src.models.pose_lstm import PoseLSTM

    assert isinstance(build_model("cnn_lstm"), CNNLSTM)
    assert isinstance(build_model("pose_lstm"), PoseLSTM)


def test_build_dataset_selects_correct_dataset(monkeypatch):
    from src.training import train

    calls = []

    class FakeVideoDataset:
        def __init__(self, **kwargs):
            calls.append(("video", kwargs))

    class FakePoseDataset:
        def __init__(self, **kwargs):
            calls.append(("pose", kwargs))

    monkeypatch.setattr(
        train,
        "KTHVideoDataset",
        FakeVideoDataset,
    )

    monkeypatch.setattr(
        train,
        "KTHPoseDataset",
        FakePoseDataset,
    )

    assert isinstance(
        build_dataset("cnn_lstm", "validation"),
        FakeVideoDataset,
    )

    assert isinstance(
        build_dataset("pose_lstm", "validation"),
        FakePoseDataset,
    )

    assert calls[0][0] == "video"
    assert calls[1][0] == "pose"