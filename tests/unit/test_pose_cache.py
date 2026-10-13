from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.features.pose_cache import (
    get_pose_cache_path,
    load_cached_pose,
    validate_pose_data,
)


def make_pose_data(num_frames=16):
    return {
        "landmarks": np.zeros(
            (num_frames, 33, 4),
            dtype=np.float32,
        ),
        "detection_mask": np.zeros(
            num_frames,
            dtype=bool,
        ),
        "frame_indices": np.arange(
            num_frames,
            dtype=np.int32,
        ),
    }


def test_validate_pose_data_accepts_valid_data():
    data = make_pose_data()

    validate_pose_data(data)


def test_validate_pose_data_rejects_wrong_shape():
    data = make_pose_data()

    data["landmarks"] = np.zeros(
        (16, 32, 4),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        validate_pose_data(data)


def test_validate_pose_data_rejects_missing_keys():
    data = make_pose_data()

    del data["detection_mask"]

    with pytest.raises(ValueError):
        validate_pose_data(data)


def test_validate_pose_data_rejects_nan():
    data = make_pose_data()

    data["landmarks"][0, 0, 0] = np.nan

    with pytest.raises(ValueError):
        validate_pose_data(data)


def test_validate_pose_data_rejects_unsorted_indices():
    data = make_pose_data()

    data["frame_indices"][5] = 2

    with pytest.raises(ValueError):
        validate_pose_data(data)


def test_cache_path_preserves_video_structure(tmp_path):
    raw_dir = tmp_path / "raw"

    video = (
        raw_dir
        / "videos"
        / "walking"
        / "example.avi"
    )

    with patch(
        "src.features.pose_cache.RAW_DATA_DIR",
        raw_dir,
    ), patch(
        "src.features.pose_cache.PROCESSED_DATA_DIR",
        tmp_path / "processed",
    ):
        path = get_pose_cache_path(video)

    assert path == (
        tmp_path
        / "processed"
        / "poses"
        / "v1_t16_full"
        / "videos"
        / "walking"
        / "example.npz"
    )


def test_cache_reuses_existing_file(tmp_path):
    raw_dir = tmp_path / "raw"
    video = raw_dir / "videos" / "walking" / "example.avi"

    video.parent.mkdir(parents=True)
    video.touch()

    expected = make_pose_data()

    with patch(
        "src.features.pose_cache.RAW_DATA_DIR",
        raw_dir,
    ), patch(
        "src.features.pose_cache.PROCESSED_DATA_DIR",
        tmp_path / "processed",
    ), patch(
        "src.features.pose_cache.extract_video_landmarks",
        return_value=expected,
    ) as extractor:

        first = load_cached_pose(video)
        second = load_cached_pose(video)

    assert extractor.call_count == 1

    np.testing.assert_array_equal(
        first["landmarks"],
        second["landmarks"],
    )

    np.testing.assert_array_equal(
        first["detection_mask"],
        second["detection_mask"],
    )

    np.testing.assert_array_equal(
        first["frame_indices"],
        second["frame_indices"],
    )


def test_cache_rejects_corrupted_data(tmp_path):
    raw_dir = tmp_path / "raw"
    video = raw_dir / "videos" / "boxing" / "example.avi"

    video.parent.mkdir(parents=True)
    video.touch()

    with patch(
        "src.features.pose_cache.RAW_DATA_DIR",
        raw_dir,
    ), patch(
        "src.features.pose_cache.PROCESSED_DATA_DIR",
        tmp_path / "processed",
    ):
        cache_path = get_pose_cache_path(video)

        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            cache_path,
            landmarks=np.zeros(
                (10, 33, 4),
                dtype=np.float32,
            ),
            detection_mask=np.zeros(
                10,
                dtype=bool,
            ),
            frame_indices=np.arange(
                10,
                dtype=np.int32,
            ),
        )

        with pytest.raises(ValueError):
            load_cached_pose(video)