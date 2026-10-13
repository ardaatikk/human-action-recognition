from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src.config.paths import RAW_DATA_DIR
from src.features.pose_extractor import PoseExtractor


def benchmark_video(
    video_path: Path,
    max_frames: int = 100,
) -> dict[str, float]:
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid FPS: {video_path}")

    initialization_time = 0.0
    inference_time = 0.0
    cleanup_time = 0.0

    processed_frames = 0
    detected_frames = 0

    try:
        start = time.perf_counter()

        extractor = PoseExtractor(running_mode="video")

        initialization_time = time.perf_counter() - start

        try:
            while processed_frames < max_frames:
                success, frame = capture.read()

                if not success:
                    break

                timestamp_ms = round(
                    processed_frames * 1000 / fps
                )

                start = time.perf_counter()

                _, detected = extractor.extract(
                    frame,
                    timestamp_ms=timestamp_ms,
                )

                inference_time += time.perf_counter() - start

                processed_frames += 1
                detected_frames += int(detected)

        finally:
            start = time.perf_counter()

            extractor.close()

            cleanup_time = time.perf_counter() - start

    finally:
        capture.release()

    if processed_frames == 0:
        raise RuntimeError("No frames processed.")

    return {
        "initialization_seconds": initialization_time,
        "inference_seconds": inference_time,
        "cleanup_seconds": cleanup_time,
        "processed_frames": processed_frames,
        "detected_frames": detected_frames,
        "inference_fps": processed_frames / inference_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MediaPipe pose extraction."
    )

    parser.add_argument(
        "--video",
        type=str,
        default="videos/boxing/person01_boxing_d1_uncomp.avi",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    if args.max_frames <= 0 or args.runs <= 0:
        parser.error("--max-frames and --runs must be positive")

    video_path = RAW_DATA_DIR / args.video

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    results = []

    for run in range(args.runs):
        result = benchmark_video(
            video_path,
            max_frames=args.max_frames,
        )

        results.append(result)

        print(f"\nRun {run + 1}/{args.runs}")
        print(
            f"Initialization: "
            f"{result['initialization_seconds']:.4f} s"
        )
        print(
            f"Inference:      "
            f"{result['inference_seconds']:.4f} s"
        )
        print(
            f"Cleanup:        "
            f"{result['cleanup_seconds']:.4f} s"
        )
        print(
            f"Inference FPS:  "
            f"{result['inference_fps']:.2f}"
        )

    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)

    for key in (
        "initialization_seconds",
        "inference_seconds",
        "cleanup_seconds",
        "inference_fps",
    ):
        values = [result[key] for result in results]

        print(
            f"{key}: "
            f"{np.mean(values):.4f} "
            f"+/- {np.std(values):.4f}"
        )


if __name__ == "__main__":
    main()