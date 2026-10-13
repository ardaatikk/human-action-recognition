from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

import src.data.pose_dataset as pose_dataset_module
from src.data.pose_dataset import KTHPoseDataset


@pytest.fixture
def mock_pose_environment(tmp_path, monkeypatch):
    """
    Create temporary split files and mock pose cache loading.
    """

    raw_dir = tmp_path / "raw"
    splits_dir = tmp_path / "splits"

    raw_dir.mkdir()
    splits_dir.mkdir()

    records = [
        {
            "path": "videos/boxing/video_001.avi",
            "action": "boxing",
        },
        {
            "path": "videos/walking/video_002.avi",
            "action": "walking",
        },
        {
            "path": "videos/running/video_003.avi",
            "action": "running",
        },
    ]

    for split in ("train", "validation", "test"):
        pd.DataFrame(records).to_csv(
            splits_dir / f"{split}.csv",
            index=False,
        )

    monkeypatch.setattr(
        pose_dataset_module,
        "RAW_DATA_DIR",
        raw_dir,
    )

    monkeypatch.setattr(
        pose_dataset_module,
        "SPLITS_DIR",
        splits_dir,
    )

    def fake_load_cached_pose(
        video_path: Path,
        num_frames: int = 16,
    ):
        landmarks = np.zeros(
            (num_frames, 33, 4),
            dtype=np.float32,
        )

        landmarks[:, :, 0] = 0.5
        landmarks[:, :, 1] = 0.4
        landmarks[:, :, 2] = -0.1
        landmarks[:, :, 3] = 0.9

        return {
            "landmarks": landmarks,
            "detection_mask": np.ones(
                num_frames,
                dtype=bool,
            ),
            "frame_indices": np.arange(
                num_frames,
                dtype=np.int32,
            ),
        }

    monkeypatch.setattr(
        pose_dataset_module,
        "load_cached_pose",
        fake_load_cached_pose,
    )

    return {
        "raw_dir": raw_dir,
        "splits_dir": splits_dir,
    }


def test_dataset_length(mock_pose_environment):
    dataset = KTHPoseDataset("train")

    assert len(dataset) == 3


def test_sample_shape(mock_pose_environment):
    dataset = KTHPoseDataset("train")

    features, label = dataset[0]

    assert features.shape == (16, 135)


def test_sample_dtype(mock_pose_environment):
    dataset = KTHPoseDataset("train")

    features, label = dataset[0]

    assert features.dtype == torch.float32


def test_label_mapping(mock_pose_environment):
    dataset = KTHPoseDataset("train")

    assert dataset[0][1] == dataset.class_to_idx["boxing"]
    assert dataset[1][1] == dataset.class_to_idx["walking"]
    assert dataset[2][1] == dataset.class_to_idx["running"]


def test_dataloader_batch(mock_pose_environment):
    dataset = KTHPoseDataset("train")

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    features, labels = next(iter(loader))

    assert features.shape == (2, 16, 135)
    assert labels.shape == (2,)


def test_custom_num_frames(mock_pose_environment):
    dataset = KTHPoseDataset(
        "train",
        num_frames=8,
    )

    features, label = dataset[0]

    assert features.shape == (8, 135)


def test_invalid_split():
    with pytest.raises(ValueError):
        KTHPoseDataset("invalid")


def test_invalid_num_frames():
    with pytest.raises(ValueError):
        KTHPoseDataset(
            "train",
            num_frames=0,
        )


def test_missing_split_file(
    mock_pose_environment,
):
    split_path = (
        mock_pose_environment["splits_dir"]
        / "train.csv"
    )

    split_path.unlink()

    with pytest.raises(FileNotFoundError):
        KTHPoseDataset("train")


def test_missing_required_column(
    mock_pose_environment,
):
    split_path = (
        mock_pose_environment["splits_dir"]
        / "train.csv"
    )

    pd.DataFrame(
        {
            "path": ["videos/example.avi"],
        }
    ).to_csv(
        split_path,
        index=False,
    )

    with pytest.raises(ValueError):
        KTHPoseDataset("train")


def test_unknown_action(
    mock_pose_environment,
):
    split_path = (
        mock_pose_environment["splits_dir"]
        / "train.csv"
    )

    pd.DataFrame(
        {
            "path": ["videos/example.avi"],
            "action": ["unknown_action"],
        }
    ).to_csv(
        split_path,
        index=False,
    )

    with pytest.raises(ValueError):
        KTHPoseDataset("train")


def test_missing_action(
    mock_pose_environment,
):
    split_path = (
        mock_pose_environment["splits_dir"]
        / "train.csv"
    )

    pd.DataFrame(
        {
            "path": ["videos/example.avi"],
            "action": [None],
        }
    ).to_csv(
        split_path,
        index=False,
    )

    with pytest.raises(ValueError):
        KTHPoseDataset("train")