from __future__ import annotations

import numpy as np


LEFT_HIP_INDEX = 23
RIGHT_HIP_INDEX = 24

NUM_LANDMARKS = 33
LANDMARK_DIMENSIONS = 4

NUM_POSE_FEATURES = 135


def normalize_pose_sequence(
    landmarks: np.ndarray,
    detection_mask: np.ndarray,
) -> np.ndarray:
    """
    Convert MediaPipe landmarks into LSTM-ready features.

    Input:
        landmarks: (T, 33, 4)
        detection_mask: (T,)

    Output:
        features: (T, 135)

    Feature layout:
        0:99    -> hip-centered x, y, z coordinates
        99:132  -> landmark visibility
        132:134 -> hip center x, y
        134     -> detection mask
    """

    landmarks = np.asarray(landmarks, dtype=np.float32)
    detection_mask = np.asarray(detection_mask)

    if landmarks.ndim != 3:
        raise ValueError(
            "landmarks must have shape (T, 33, 4)"
        )

    if landmarks.shape[1:] != (
        NUM_LANDMARKS,
        LANDMARK_DIMENSIONS,
    ):
        raise ValueError(
            f"Expected landmarks shape (T, 33, 4), "
            f"got {landmarks.shape}"
        )

    num_frames = landmarks.shape[0]

    if detection_mask.shape != (num_frames,):
        raise ValueError(
            f"Expected detection_mask shape ({num_frames},), "
            f"got {detection_mask.shape}"
        )

    if not np.isfinite(landmarks).all():
        raise ValueError(
            "landmarks contain NaN or infinite values"
        )

    if not np.isin(detection_mask, [0, 1]).all():
        raise ValueError(
            "detection_mask must contain only 0 or 1"
        )

    detection_mask = detection_mask.astype(bool)

    features = np.zeros(
        (num_frames, NUM_POSE_FEATURES),
        dtype=np.float32,
    )

    for frame_index in range(num_frames):

        if not detection_mask[frame_index]:
            continue

        frame_landmarks = landmarks[frame_index]

        coordinates = frame_landmarks[:, :3]

        visibility = frame_landmarks[:, 3]

        left_hip = coordinates[LEFT_HIP_INDEX]
        right_hip = coordinates[RIGHT_HIP_INDEX]

        hip_center = (left_hip + right_hip) / 2.0

        centered_coordinates = coordinates - hip_center

        features[frame_index, 0:99] = (
            centered_coordinates.reshape(-1)
        )

        features[frame_index, 99:132] = visibility

        features[frame_index, 132:134] = hip_center[:2]

        features[frame_index, 134] = 1.0

    return features