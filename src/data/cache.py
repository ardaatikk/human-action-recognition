from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from src.config.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, SPLITS_DIR
from src.config.settings import FRAME_HEIGHT, FRAME_WIDTH, NUM_FRAMES
from src.data.preprocess import load_video_frames


CACHE_VERSION = "v1"


def get_cache_dir(
    num_frames: int = NUM_FRAMES,
    height: int = FRAME_HEIGHT,
    width: int = FRAME_WIDTH,
) -> Path:
    return (
        PROCESSED_DATA_DIR
        / "frames"
        / f"{CACHE_VERSION}_t{num_frames}_h{height}_w{width}"
    )


def get_cache_path(
    video_path: Path,
    *,
    num_frames: int = NUM_FRAMES,
    height: int = FRAME_HEIGHT,
    width: int = FRAME_WIDTH,
) -> Path:
    relative_path = video_path.resolve().relative_to(
        RAW_DATA_DIR.resolve()
    )

    return (
        get_cache_dir(num_frames, height, width)
        / relative_path.with_suffix(".npy")
    )


def load_cached_frames(
    video_path: Path,
    *,
    num_frames: int = NUM_FRAMES,
    height: int = FRAME_HEIGHT,
    width: int = FRAME_WIDTH,
) -> np.ndarray:
    cache_path = get_cache_path(
        video_path,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    if cache_path.exists():
        return np.load(cache_path, allow_pickle=False)

    frames = load_video_frames(
        video_path,
        num_frames=num_frames,
    )

    expected_shape = (num_frames, height, width)

    if frames.shape != expected_shape:
        raise ValueError(
            f"Unexpected shape for {video_path}: "
            f"{frames.shape} != {expected_shape}"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = cache_path.with_name(
        cache_path.stem + ".tmp.npy"
    )

    try:
        np.save(temporary_path, frames)
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute cached KTH video frames."
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "validation", "test"],
        default=["train", "validation", "test"],
    )

    args = parser.parse_args()

    import pandas as pd

    paths: set[Path] = set()

    for split in args.splits:
        csv_path = SPLITS_DIR / f"{split}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        dataframe = pd.read_csv(csv_path)

        for relative_path in dataframe["path"]:
            paths.add(RAW_DATA_DIR / relative_path)

    cached_count = 0
    created_count = 0

    for video_path in tqdm(
        sorted(paths),
        desc="Caching videos",
        dynamic_ncols=True,
    ):
        cache_path = get_cache_path(video_path)

        if cache_path.exists():
            cached_count += 1
            continue

        load_cached_frames(video_path)
        created_count += 1

    print("\n[CACHE COMPLETE]")
    print(f"Created: {created_count}")
    print(f"Already cached: {cached_count}")
    print(f"Total: {len(paths)}")
    print(f"Directory: {get_cache_dir()}")


if __name__ == "__main__":
    main()