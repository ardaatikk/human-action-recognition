import numpy as np
import pytest

from src.training.metrics import calculate_classification_metrics


CLASS_NAMES = ["boxing", "walking", "running"]


def test_perfect_predictions():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])

    metrics = calculate_classification_metrics(
        y_true,
        y_pred,
        CLASS_NAMES,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision_macro"] == 1.0
    assert metrics["recall_macro"] == 1.0
    assert metrics["f1_macro"] == 1.0


def test_partially_correct_predictions():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])

    metrics = calculate_classification_metrics(
        y_true,
        y_pred,
        CLASS_NAMES,
    )

    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["f1_macro"] == pytest.approx(2 / 3)


def test_confusion_matrix():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])

    metrics = calculate_classification_metrics(
        y_true,
        y_pred,
        CLASS_NAMES,
    )

    expected = np.array([
        [2, 0, 0],
        [0, 1, 1],
        [0, 1, 1],
    ])

    np.testing.assert_array_equal(
        metrics["confusion_matrix"],
        expected,
    )


def test_classification_report_contains_all_classes():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])

    metrics = calculate_classification_metrics(
        y_true,
        y_pred,
        CLASS_NAMES,
    )

    report = metrics["classification_report"]

    for class_name in CLASS_NAMES:
        assert class_name in report
        assert report[class_name]["f1-score"] == 1.0


def test_missing_predicted_class():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 0, 0])

    metrics = calculate_classification_metrics(
        y_true,
        y_pred,
        CLASS_NAMES,
    )

    assert metrics["confusion_matrix"].shape == (3, 3)
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["precision_macro"] >= 0
    assert metrics["f1_macro"] >= 0


def test_mismatched_shapes():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1])

    with pytest.raises(ValueError, match="same shape"):
        calculate_classification_metrics(
            y_true,
            y_pred,
            CLASS_NAMES,
        )


def test_multidimensional_predictions():
    y_true = np.array([[0, 1], [1, 2]])
    y_pred = np.array([[0, 1], [1, 2]])

    with pytest.raises(ValueError, match="one-dimensional"):
        calculate_classification_metrics(
            y_true,
            y_pred,
            CLASS_NAMES,
        )


def test_empty_predictions():
    y_true = np.array([])
    y_pred = np.array([])

    with pytest.raises(ValueError, match="empty"):
        calculate_classification_metrics(
            y_true,
            y_pred,
            CLASS_NAMES,
        )


def test_metrics_return_expected_keys():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])

    metrics = calculate_classification_metrics(
        y_true,
        y_pred,
        CLASS_NAMES,
    )

    expected_keys = {
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "confusion_matrix",
        "classification_report",
    }

    assert set(metrics.keys()) == expected_keys