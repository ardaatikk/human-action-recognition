from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, SPLITS_DIR
from src.config.settings import NUM_FRAMES
from src.features.video_pose import extract_video_landmarks


CACHE_VERSION = "v1"
POSE_MODEL_VARIANT = "full"

VALID_SPLITS = ("train", "validation", "test")


def get_pose_cache_dir(
    num_frames: int = NUM_FRAMES,
) -> Path:
    return (
        PROCESSED_DATA_DIR
        / "poses"
        / f"{CACHE_VERSION}_t{num_frames}_{POSE_MODEL_VARIANT}"
    )


def get_pose_cache_path(
    video_path: Path,
    num_frames: int = NUM_FRAMES,
) -> Path:
    video_path = Path(video_path).resolve()
    raw_dir = RAW_DATA_DIR.resolve()

    try:
        relative_path = video_path.relative_to(raw_dir)
    except ValueError as exc:
        raise ValueError(
            f"Video must be inside {raw_dir}: {video_path}"
        ) from exc

    return (
        get_pose_cache_dir(num_frames)
        / relative_path.with_suffix(".npz")
    )


def validate_pose_data(
    data: dict[str, np.ndarray],
    num_frames: int = NUM_FRAMES,
) -> None:
    required_keys = {
        "landmarks",
        "detection_mask",
        "frame_indices",
    }

    if set(data) != required_keys:
        raise ValueError(
            f"Unexpected pose data keys: {set(data)}"
        )

    if data["landmarks"].shape != (num_frames, 33, 4):
        raise ValueError(
            f"Invalid landmarks shape: {data['landmarks'].shape}"
        )

    if data["detection_mask"].shape != (num_frames,):
        raise ValueError(
            "Invalid detection_mask shape."
        )

    if data["frame_indices"].shape != (num_frames,):
        raise ValueError(
            "Invalid frame_indices shape."
        )

    if not np.isfinite(data["landmarks"]).all():
        raise ValueError(
            "Landmarks contain NaN or infinite values."
        )

    if not np.all(np.diff(data["frame_indices"]) >= 0):
        raise ValueError(
            "Frame indices must be non-decreasing."
        )


def load_cached_pose(
    video_path: Path,
    num_frames: int = NUM_FRAMES,
) -> dict[str, np.ndarray]:
    cache_path = get_pose_cache_path(
        video_path,
        num_frames=num_frames,
    )

    if cache_path.exists():
        with np.load(
            cache_path,
            allow_pickle=False,
        ) as archive:
            data = {
                key: archive[key]
                for key in (
                    "landmarks",
                    "detection_mask",
                    "frame_indices",
                )
            }

        validate_pose_data(data, num_frames)
        return data

    data = extract_video_landmarks(
        video_path,
        num_frames=num_frames,
    )

    validate_pose_data(data, num_frames)

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = cache_path.with_suffix(".tmp.npz")

    try:
        np.savez_compressed(
            temporary_path,
            **data,
        )

        os.replace(
            temporary_path,
            cache_path,
        )
    finally:
        temporary_path.unlink(missing_ok=True)

    return data


def cache_split(
    split: str,
    num_frames: int = NUM_FRAMES,
) -> None:
    if split not in VALID_SPLITS:
        raise ValueError(
            f"Invalid split: {split}"
        )

    split_path = SPLITS_DIR / f"{split}.csv"

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}"
        )

    dataframe = pd.read_csv(split_path)

    for relative_path in tqdm(
        dataframe["path"],
        desc=f"Caching poses [{split}]",
    ):
        video_path = RAW_DATA_DIR / relative_path

        load_cached_pose(
            video_path,
            num_frames=num_frames,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute MediaPipe pose landmarks."
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        choices=VALID_SPLITS,
        default=list(VALID_SPLITS),
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=NUM_FRAMES,
    )

    args = parser.parse_args()

    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")

    for split in args.splits:
        cache_split(
            split,
            num_frames=args.num_frames,
        )

    print("\nPose caching completed.")
    print(f"Cache directory: {get_pose_cache_dir(args.num_frames)}")


if __name__ == "__main__":
    main()