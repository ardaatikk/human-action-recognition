from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config.settings import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    NUM_FRAMES,
)


def sample_frame_indices(
    total_frames: int,
    num_frames: int = NUM_FRAMES,
) -> np.ndarray:
    """Sample temporally ordered frame indices."""

    if total_frames <= 0:
        raise ValueError("Video contains no frames.")

    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")

    indices = np.linspace(
        0,
        total_frames - 1,
        num=num_frames,
    )

    return np.rint(indices).astype(np.int64)


def preprocess_frame(
    frame: np.ndarray,
    height: int = FRAME_HEIGHT,
    width: int = FRAME_WIDTH,
) -> np.ndarray:
    """Convert a frame to normalized grayscale."""

    if frame.ndim == 3:
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )
    elif frame.ndim != 2:
        raise ValueError("Invalid frame dimensions.")

    frame = cv2.resize(
        frame,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )

    frame = frame.astype(np.float32) / 255.0

    return frame


def load_video_frames(
    video_path: Path,
    num_frames: int = NUM_FRAMES,
) -> np.ndarray:
    """Load video frames sequentially and sample a fixed-length sequence."""

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open video: {video_path}"
        )

    decoded_frames = []

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            processed_frame = preprocess_frame(frame)
            decoded_frames.append(processed_frame)

    finally:
        capture.release()

    if not decoded_frames:
        raise RuntimeError(
            f"No readable frames found in video: {video_path}"
        )

    indices = sample_frame_indices(
        total_frames=len(decoded_frames),
        num_frames=num_frames,
    )

    sampled_frames = [
        decoded_frames[index]
        for index in indices
    ]

    return np.stack(sampled_frames, axis=0)