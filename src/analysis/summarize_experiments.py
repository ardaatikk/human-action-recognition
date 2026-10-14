"""Summarize validation metrics across multiple random seeds."""

import json
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"

EXPERIMENTS = {
    "Coordinates-only LSTM": {
        42: "pose_coordinates_only_ablation",
        123: "pose_coordinates_seed123",
        456: "pose_coordinates_seed456",
    },
    "Mean-Pooled Pose MLP": {
        42: "pose_mean_mlp_baseline",
        123: "pose_mean_mlp_seed123",
        456: "pose_mean_mlp_seed456",
    },
}


def load_metrics(experiment: str) -> dict:
    """Load validation metrics for a given experiment."""
    path = (
        REPORTS_DIR
        / experiment
        / "validation"
        / "best"
        / "metrics.json"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Metrics file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    for key in ("accuracy", "f1_macro"):
        if key not in metrics:
            raise KeyError(
                f"Missing metric '{key}' in {path}"
            )

    return metrics


def summarize(values: list[float]) -> tuple[float, float]:
    """Return mean and sample standard deviation."""
    if len(values) < 2:
        raise ValueError(
            "At least two runs are required to calculate "
            "sample standard deviation."
        )

    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    print("\nMULTI-SEED VALIDATION RESULTS")
    print("=" * 76)

    print(
        f"{'Model':<26}"
        f"{'Seed':<8}"
        f"{'Accuracy':<14}"
        f"{'Macro F1':<12}"
    )
    print("-" * 76)

    summaries = {}

    for model_name, experiments in EXPERIMENTS.items():
        accuracies = []
        f1_scores = []

        for seed, experiment in experiments.items():
            metrics = load_metrics(experiment)

            accuracy = float(metrics["accuracy"])
            f1_macro = float(metrics["f1_macro"])

            accuracies.append(accuracy)
            f1_scores.append(f1_macro)

            print(
                f"{model_name:<26}"
                f"{seed:<8}"
                f"{accuracy * 100:>7.2f}%      "
                f"{f1_macro:>7.4f}"
            )

        summaries[model_name] = {
            "accuracy": summarize(accuracies),
            "f1_macro": summarize(f1_scores),
        }

    print("\nMULTI-SEED SUMMARY")
    print("=" * 76)

    print(
        f"{'Model':<26}"
        f"{'Accuracy (mean ± std)':<26}"
        f"{'Macro F1 (mean ± std)'}"
    )
    print("-" * 76)

    for model_name, metrics in summaries.items():
        accuracy_mean, accuracy_std = metrics["accuracy"]
        f1_mean, f1_std = metrics["f1_macro"]

        print(
            f"{model_name:<26}"
            f"{accuracy_mean * 100:.2f}% ± "
            f"{accuracy_std * 100:.2f} pp"
            f"{'':<8}"
            f"{f1_mean:.4f} ± {f1_std:.4f}"
        )


if __name__ == "__main__":
    main()