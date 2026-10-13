from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.config.paths import PROJECT_ROOT, RAW_DATA_DIR
from src.config.settings import NUM_FRAMES
from src.features.pose_extractor import PoseExtractor


POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]


def sample_video_frames(
    video_path: Path,
    num_frames: int = NUM_FRAMES,
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            frames.append(frame)

    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames found: {video_path}")

    indices = np.linspace(
        0,
        len(frames) - 1,
        num_frames,
        dtype=int,
    )

    return [frames[index] for index in indices]


def draw_pose(
    frame: np.ndarray,
    landmarks: np.ndarray,
    visibility_threshold: float = 0.5,
) -> np.ndarray:
    image = frame.copy()

    height, width = image.shape[:2]

    for start, end in POSE_CONNECTIONS:
        first = landmarks[start]
        second = landmarks[end]

        if (
            first[3] < visibility_threshold
            or second[3] < visibility_threshold
        ):
            continue

        p1 = (
            int(first[0] * width),
            int(first[1] * height),
        )

        p2 = (
            int(second[0] * width),
            int(second[1] * height),
        )

        cv2.line(image, p1, p2, (0, 255, 0), 2)

    for landmark in landmarks:
        if landmark[3] < visibility_threshold:
            continue

        x = int(landmark[0] * width)
        y = int(landmark[1] * height)

        cv2.circle(
            image,
            (x, y),
            3,
            (0, 0, 255),
            -1,
        )

    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize MediaPipe pose detection on KTH videos."
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    video_path = args.video

    if not video_path.is_absolute():
        video_path = RAW_DATA_DIR / video_path

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    frames = sample_video_frames(video_path)

    visualizations = []
    detected_count = 0

    with PoseExtractor(running_mode="image") as extractor:
        for frame in frames:
            landmarks, detected = extractor.extract(frame)

            if detected:
                detected_count += 1
                visualization = draw_pose(frame, landmarks)
            else:
                visualization = frame.copy()

            visualizations.append(
                cv2.cvtColor(
                    visualization,
                    cv2.COLOR_BGR2RGB,
                )
            )

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(14, 12),
    )

    for index, (axis, image) in enumerate(
        zip(axes.flat, visualizations)
    ):
        axis.imshow(image)
        axis.set_title(f"Frame {index + 1}")
        axis.axis("off")

    fig.suptitle(
        f"MediaPipe Pose Detection — {detected_count}/{NUM_FRAMES} frames",
        fontsize=16,
    )

    fig.tight_layout()

    output_dir = (
        PROJECT_ROOT
        / "artifacts"
        / "figures"
        / "pose_detection"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{video_path.stem}_pose.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)

    print("=" * 60)
    print("MEDIAPIPE POSE DETECTION")
    print("=" * 60)
    print(f"Video: {video_path.name}")
    print(f"Frames: {NUM_FRAMES}")
    print(f"Detected: {detected_count}")
    print(f"Detection rate: {detected_count / NUM_FRAMES:.2%}")
    print(f"Figure: {output_path}")


if __name__ == "__main__":
    main()