from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from tqdm import tqdm

from src.config.paths import PROJECT_ROOT


MODEL_URLS = {
    "lite": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/"
        "pose_landmarker_lite.task"
    ),
    "full": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/1/"
        "pose_landmarker_full.task"
    ),
    "heavy": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_heavy/float16/1/"
        "pose_landmarker_heavy.task"
    ),
}


MODEL_DIR = PROJECT_ROOT / "artifacts" / "models" / "mediapipe"


class DownloadProgress:
    def __init__(self) -> None:
        self.progress = None

    def __call__(
        self,
        block_count: int,
        block_size: int,
        total_size: int,
    ) -> None:
        if self.progress is None:
            self.progress = tqdm(
                total=total_size if total_size > 0 else None,
                unit="B",
                unit_scale=True,
                desc="Downloading model",
            )

        downloaded = block_count * block_size

        if self.progress.total is not None:
            downloaded = min(downloaded, self.progress.total)

        self.progress.update(
            max(0, downloaded - self.progress.n)
        )

    def close(self) -> None:
        if self.progress is not None:
            self.progress.close()


def download_pose_model(
    variant: str = "full",
    force: bool = False,
) -> Path:
    if variant not in MODEL_URLS:
        raise ValueError(
            f"Unknown model variant: {variant}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    destination = MODEL_DIR / f"pose_landmarker_{variant}.task"

    if destination.exists() and not force:
        print(f"[SKIP] Model already exists: {destination}")
        return destination

    temporary_path = destination.with_name(
        destination.name + ".tmp"
    )

    progress = DownloadProgress()

    try:
        urllib.request.urlretrieve(
            MODEL_URLS[variant],
            filename=temporary_path,
            reporthook=progress,
        )

        if temporary_path.stat().st_size == 0:
            raise RuntimeError("Downloaded model is empty.")

        temporary_path.replace(destination)

    finally:
        progress.close()
        temporary_path.unlink(missing_ok=True)

    print(f"[SUCCESS] Model downloaded: {destination}")

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MediaPipe Pose Landmarker models."
    )

    parser.add_argument(
        "--variant",
        choices=MODEL_URLS.keys(),
        default="full",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    download_pose_model(
        variant=args.variant,
        force=args.force,
    )


if __name__ == "__main__":
    main()