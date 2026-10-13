import numpy as np
import pytest

from src.features.pose_normalization import (
    NUM_POSE_FEATURES,
    normalize_pose_sequence,
)


def create_sample_landmarks(
    num_frames: int = 4,
) -> np.ndarray:
    """Create synthetic landmarks for testing."""

    landmarks = np.zeros(
        (num_frames, 33, 4),
        dtype=np.float32,
    )

    landmarks[:, :, 0] = 0.5
    landmarks[:, :, 1] = 0.4
    landmarks[:, :, 2] = -0.1
    landmarks[:, :, 3] = 0.9

    return landmarks


def test_output_shape():
    landmarks = create_sample_landmarks()
    mask = np.ones(4, dtype=bool)

    features = normalize_pose_sequence(landmarks, mask)

    assert features.shape == (4, NUM_POSE_FEATURES)


def test_output_dtype():
    landmarks = create_sample_landmarks()
    mask = np.ones(4, dtype=bool)

    features = normalize_pose_sequence(landmarks, mask)

    assert features.dtype == np.float32


def test_hip_centering():
    landmarks = create_sample_landmarks(1)

    # Hip coordinates
    landmarks[0, 23, :3] = [0.4, 0.6, -0.2]
    landmarks[0, 24, :3] = [0.6, 0.6, -0.2]

    # Shoulder coordinate
    landmarks[0, 11, :3] = [0.3, 0.3, -0.1]

    mask = np.array([True])

    features = normalize_pose_sequence(landmarks, mask)

    centered = features[0, :99].reshape(33, 3)

    expected_shoulder = [-0.2, -0.3, 0.1]

    np.testing.assert_allclose(
        centered[11],
        expected_shoulder,
        atol=1e-6,
    )


def test_hip_center_position_preserved():
    landmarks = create_sample_landmarks(1)

    landmarks[0, 23, :3] = [0.4, 0.6, -0.2]
    landmarks[0, 24, :3] = [0.6, 0.6, -0.2]

    mask = np.array([True])

    features = normalize_pose_sequence(landmarks, mask)

    np.testing.assert_allclose(
        features[0, 132:134],
        [0.5, 0.6],
        atol=1e-6,
    )


def test_visibility_preserved():
    landmarks = create_sample_landmarks(1)
    landmarks[0, 11, 3] = 0.75

    mask = np.array([True])

    features = normalize_pose_sequence(landmarks, mask)

    assert features[0, 99 + 11] == pytest.approx(0.75)


def test_missing_frames_are_zero():
    landmarks = create_sample_landmarks()
    mask = np.array([True, False, True, False])

    features = normalize_pose_sequence(landmarks, mask)

    assert np.all(features[1] == 0)
    assert np.all(features[3] == 0)


def test_detection_mask_feature():
    landmarks = create_sample_landmarks()
    mask = np.array([True, False, True, False])

    features = normalize_pose_sequence(landmarks, mask)

    np.testing.assert_array_equal(
        features[:, 134],
        [1, 0, 1, 0],
    )


def test_invalid_landmark_shape():
    landmarks = np.zeros((4, 20, 4))
    mask = np.ones(4, dtype=bool)

    with pytest.raises(ValueError):
        normalize_pose_sequence(landmarks, mask)


def test_invalid_mask_shape():
    landmarks = create_sample_landmarks()
    mask = np.ones(3, dtype=bool)

    with pytest.raises(ValueError):
        normalize_pose_sequence(landmarks, mask)


def test_nan_values_rejected():
    landmarks = create_sample_landmarks()
    landmarks[0, 0, 0] = np.nan

    mask = np.ones(4, dtype=bool)

    with pytest.raises(ValueError):
        normalize_pose_sequence(landmarks, mask)


def test_infinite_values_rejected():
    landmarks = create_sample_landmarks()
    landmarks[0, 0, 0] = np.inf

    mask = np.ones(4, dtype=bool)

    with pytest.raises(ValueError):
        normalize_pose_sequence(landmarks, mask)


def test_invalid_mask_values():
    landmarks = create_sample_landmarks()
    mask = np.array([1, 0, 2, 1])

    with pytest.raises(ValueError):
        normalize_pose_sequence(landmarks, mask)


def test_all_frames_missing():
    landmarks = create_sample_landmarks()
    mask = np.zeros(4, dtype=bool)

    features = normalize_pose_sequence(landmarks, mask)

    assert np.all(features == 0)


def test_all_frames_detected():
    landmarks = create_sample_landmarks()
    mask = np.ones(4, dtype=bool)

    features = normalize_pose_sequence(landmarks, mask)

    assert np.all(features[:, 134] == 1)


def test_input_not_modified():
    landmarks = create_sample_landmarks()
    mask = np.ones(4, dtype=bool)

    original = landmarks.copy()

    normalize_pose_sequence(landmarks, mask)

    np.testing.assert_array_equal(
        landmarks,
        original,
    )


def test_empty_sequence():
    landmarks = np.zeros((0, 33, 4), dtype=np.float32)
    mask = np.zeros(0, dtype=bool)

    features = normalize_pose_sequence(landmarks, mask)

    assert features.shape == (0, NUM_POSE_FEATURES)