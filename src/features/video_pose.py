from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config.settings import NUM_FRAMES
from src.features.pose_extractor import PoseExtractor


def sample_frame_indices(
    total_frames: int,
    num_frames: int = NUM_FRAMES,
) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive.")

    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")

    return np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int,
    )


def extract_video_landmarks(
    video_path: Path,
    num_frames: int = NUM_FRAMES,
) -> dict[str, np.ndarray]:
    """
    Extract pose landmarks from a complete video.

    The full video is processed sequentially in VIDEO mode.
    Uniformly sampled frames are then selected for the model.
    """

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError(
            f"Invalid FPS for video: {video_path}"
        )

    all_landmarks = []
    all_detection_flags = []

    frame_index = 0

    try:
        with PoseExtractor(running_mode="video") as extractor:
            while True:
                success, frame = capture.read()

                if not success:
                    break

                timestamp_ms = round(
                    frame_index * 1000 / fps
                )

                landmarks, detected = extractor.extract(
                    frame,
                    timestamp_ms=timestamp_ms,
                )

                all_landmarks.append(landmarks)
                all_detection_flags.append(detected)

                frame_index += 1

    finally:
        capture.release()

    if not all_landmarks:
        raise RuntimeError(
            f"No frames decoded from video: {video_path}"
        )

    indices = sample_frame_indices(
        total_frames=len(all_landmarks),
        num_frames=num_frames,
    )

    landmarks = np.stack(
        all_landmarks,
        axis=0,
    )[indices]

    detection_mask = np.asarray(
        all_detection_flags,
        dtype=bool,
    )[indices]

    return {
        "landmarks": landmarks.astype(np.float32),
        "detection_mask": detection_mask,
        "frame_indices": indices.astype(np.int32),
    }