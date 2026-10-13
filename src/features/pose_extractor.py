from __future__ import annotations
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from src.config.paths import PROJECT_ROOT


POSE_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "mediapipe"
    / "pose_landmarker_full.task"
)

NUM_LANDMARKS = 33
NUM_FEATURES = 4


class PoseExtractor:
    def __init__(
        self,
        model_path: Path = POSE_MODEL_PATH,
        running_mode: str = "video",
    ):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found: {model_path}\n"
                "Run: python -m src.data.download_pose_model --variant full"
            )

        if running_mode not in {"image", "video"}:
            raise ValueError(
                "running_mode must be 'image' or 'video'."
            )

        self.running_mode = running_mode
        self.last_timestamp_ms = -1

        mode = (
            mp.tasks.vision.RunningMode.VIDEO
            if running_mode == "video"
            else mp.tasks.vision.RunningMode.IMAGE
        )

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path),
            ),
            running_mode=mode,
            num_poses=1,
        )

        self.landmarker = (
            mp.tasks.vision.PoseLandmarker.create_from_options(options)
        )

    def extract(
        self,
        frame: np.ndarray,
        timestamp_ms: int | None = None,
    ) -> tuple[np.ndarray, bool]:
        """
        Extract 33 pose landmarks from a BGR frame.

        Returns:
            landmarks: (33, 4) float32 array.
            detected: Whether a pose was detected.
        """

        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "Expected a BGR image with shape (H, W, 3)."
            )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        rgb_frame = np.ascontiguousarray(rgb_frame)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        if self.running_mode == "video":
            if timestamp_ms is None:
                raise ValueError(
                    "timestamp_ms is required in VIDEO mode."
                )

            if timestamp_ms <= self.last_timestamp_ms:
                raise ValueError(
                    "VIDEO timestamps must be strictly increasing."
                )

            result = self.landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            self.last_timestamp_ms = timestamp_ms

        else:
            result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return (
                np.zeros(
                    (NUM_LANDMARKS, NUM_FEATURES),
                    dtype=np.float32,
                ),
                False,
            )

        landmarks = result.pose_landmarks[0]

        features = np.array(
            [
                [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility,
                ]
                for landmark in landmarks
            ],
            dtype=np.float32,
        )

        return features, True

    def close(self) -> None:
        self.landmarker.close()

    def __enter__(self) -> "PoseExtractor":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()