from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.paths import RAW_DATA_DIR, REPORTS_DIR, SPLITS_DIR
from src.features.pose_cache import get_pose_cache_path


DETECTION_BINS = [-0.001, 1 / 3, 2 / 3, 1.001]
DETECTION_LABELS = ["Low", "Medium", "High"]


def load_analysis_data(
    experiment: str,
    split: str,
    checkpoint: str,
) -> pd.DataFrame:
    split_path = SPLITS_DIR / f"{split}.csv"

    predictions_path = (
        REPORTS_DIR
        / experiment
        / split
        / checkpoint
        / "predictions.csv"
    )

    if not split_path.exists():
        raise FileNotFoundError(split_path)

    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)

    videos = pd.read_csv(split_path)
    predictions = pd.read_csv(predictions_path)

    if len(videos) != len(predictions):
        raise ValueError(
            "Split and predictions contain different numbers of rows."
        )

    if not (
        videos["action"].to_numpy()
        == predictions["true_class"].to_numpy()
    ).all():
        raise ValueError(
            "Prediction order does not match the dataset split."
        )

    dataframe = videos[["path", "action"]].copy()

    dataframe["predicted_class"] = predictions["predicted_class"].to_numpy()
    dataframe["confidence"] = predictions["confidence"].to_numpy()

    dataframe["correct"] = (
        dataframe["action"] == dataframe["predicted_class"]
    )

    detection_rates = []

    for relative_path in dataframe["path"]:
        video_path = RAW_DATA_DIR / relative_path
        cache_path = get_pose_cache_path(video_path)

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing pose cache: {cache_path}"
            )

        with np.load(cache_path, allow_pickle=False) as archive:
            detection_mask = archive["detection_mask"]

        detection_rates.append(
            float(np.mean(detection_mask))
        )

    dataframe["detection_rate"] = detection_rates

    dataframe["detection_group"] = pd.cut(
        dataframe["detection_rate"],
        bins=DETECTION_BINS,
        labels=DETECTION_LABELS,
        include_lowest=True,
    )

    return dataframe


def summarize_by_detection_group(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        dataframe.groupby(
            "detection_group",
            observed=False,
        )
        .agg(
            videos=("correct", "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
            mean_detection_rate=("detection_rate", "mean"),
        )
        .reset_index()
    )

    return summary


def summarize_by_class(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        dataframe.groupby("action")
        .agg(
            videos=("correct", "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
            mean_detection_rate=("detection_rate", "mean"),
        )
        .reset_index()
    )

    return summary


def summarize_class_and_group(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    return (
        dataframe.groupby(
            ["action", "detection_group"],
            observed=True,
        )
        .agg(
            videos=("correct", "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )


def plot_detection_impact(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    positions = np.arange(len(summary))

    ax.bar(
        positions,
        summary["accuracy"] * 100,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(summary["detection_group"])

    ax.set_ylim(0, 100)
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_xlabel("Pose Detection Rate Group")
    ax.set_title("Pose Detection Rate vs Classification Accuracy")

    for position, row in enumerate(summary.itertuples()):
        if row.videos == 0:
            label = "n=0"
        else:
            label = f"{row.accuracy:.1%}\n(n={row.videos})"

        ax.text(
            position,
            3 if row.videos == 0 else row.accuracy * 100 + 2,
            label,
            ha="center",
        )

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze pose detection impact on predictions."
    )

    parser.add_argument(
        "--experiment",
        default="pose_lstm_baseline",
    )

    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="validation",
    )

    parser.add_argument(
        "--checkpoint",
        choices=["best", "last"],
        default="best",
    )

    args = parser.parse_args()

    dataframe = load_analysis_data(
        experiment=args.experiment,
        split=args.split,
        checkpoint=args.checkpoint,
    )

    output_dir = (
        REPORTS_DIR
        / args.experiment
        / args.split
        / args.checkpoint
        / "pose_detection_impact"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    group_summary = summarize_by_detection_group(dataframe)
    class_summary = summarize_by_class(dataframe)
    class_group_summary = summarize_class_and_group(dataframe)

    dataframe.to_csv(
        output_dir / "video_analysis.csv",
        index=False,
    )

    group_summary.to_csv(
        output_dir / "detection_group_summary.csv",
        index=False,
    )

    class_summary.to_csv(
        output_dir / "class_summary.csv",
        index=False,
    )

    class_group_summary.to_csv(
        output_dir / "class_detection_summary.csv",
        index=False,
    )

    plot_detection_impact(
        group_summary,
        output_dir / "detection_impact.png",
    )

    print("\nPOSE DETECTION IMPACT ANALYSIS")
    print("=" * 55)

    print(f"\nVideos analyzed: {len(dataframe)}")
    print(f"Overall accuracy: {dataframe['correct'].mean():.2%}")

    print("\nAccuracy by detection group:")
    print(group_summary.to_string(index=False))

    print("\nAccuracy by action:")
    print(class_summary.to_string(index=False))

    print("\nAccuracy by action and detection group:")
    print(class_group_summary.to_string(index=False))

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()