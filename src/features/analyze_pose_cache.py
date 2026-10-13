from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config.paths import (
    RAW_DATA_DIR,
    REPORTS_DIR,
    SPLITS_DIR,
)
from src.features.pose_cache import (
    get_pose_cache_path,
    validate_pose_data,
)


VALID_SPLITS = ("train", "validation", "test")


def analyze_split(split: str) -> pd.DataFrame:
    split_file = SPLITS_DIR / f"{split}.csv"

    if not split_file.exists():
        raise FileNotFoundError(split_file)

    dataframe = pd.read_csv(split_file)

    records = []

    for _, row in dataframe.iterrows():
        video_path = RAW_DATA_DIR / row["path"]
        cache_path = get_pose_cache_path(video_path)

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing pose cache: {cache_path}"
            )

        with np.load(cache_path, allow_pickle=False) as archive:
            pose_data = {
                "landmarks": archive["landmarks"],
                "detection_mask": archive["detection_mask"],
                "frame_indices": archive["frame_indices"],
            }

        validate_pose_data(pose_data)

        mask = pose_data["detection_mask"]

        records.append(
            {
                "split": split,
                "action": row["action"],
                "video": video_path.name,
                "total_frames": len(mask),
                "detected_frames": int(mask.sum()),
                "detection_rate": float(mask.mean()),
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze cached MediaPipe pose detection quality."
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        choices=VALID_SPLITS,
        default=list(VALID_SPLITS),
    )

    args = parser.parse_args()

    results = pd.concat(
        [analyze_split(split) for split in args.splits],
        ignore_index=True,
    )

    summary = (
        results.groupby(["split", "action"])
        .agg(
            videos=("video", "count"),
            mean_detection_rate=("detection_rate", "mean"),
            min_detection_rate=("detection_rate", "min"),
            max_detection_rate=("detection_rate", "max"),
            fully_detected_videos=(
                "detection_rate",
                lambda values: int((values == 1.0).sum()),
            ),
            zero_detection_videos=(
                "detection_rate",
                lambda values: int((values == 0.0).sum()),
            ),
        )
        .reset_index()
    )

    summary["mean_detection_rate"] *= 100
    summary["min_detection_rate"] *= 100
    summary["max_detection_rate"] *= 100

    output_dir = REPORTS_DIR / "pose_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        output_dir / "video_detection_rates.csv",
        index=False,
    )

    summary.to_csv(
        output_dir / "class_detection_summary.csv",
        index=False,
    )

    print("\n" + "=" * 75)
    print("POSE DETECTION QUALITY ANALYSIS")
    print("=" * 75)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    print("\nOverall detection rate:")
    print(f"{results['detection_rate'].mean():.2%}")

    print("\nVideos with zero detections:")
    print(int((results["detection_rate"] == 0).sum()))

    print("\nVideos with full detections:")
    print(int((results["detection_rate"] == 1).sum()))

    print(f"\nReports saved to: {output_dir}")


if __name__ == "__main__":
    main()