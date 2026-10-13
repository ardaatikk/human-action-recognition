from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | tuple[str, ...],
) -> dict:
    """
    Calculate classification metrics for a multi-class problem.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    if y_true.ndim != 1:
        raise ValueError(
            "y_true and y_pred must be one-dimensional."
        )

    if y_true.size == 0:
        raise ValueError(
            "Cannot calculate metrics for empty predictions."
        )

    labels = list(range(len(class_names)))

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    return metrics