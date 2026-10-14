import numpy as np
import pytest
import torch

from src.config.settings import ACTION_CLASSES
from src.training.train import build_dataset, build_model


@pytest.mark.parametrize(
    "model_name, expected_features",
    [
        ("pose_lstm", 135),
        ("pose_mask_lstm", 1),
        ("pose_coordinates_lstm", 99),
        ("pose_mean_mlp", 99),
    ],
)
def test_pose_model_dataset_compatibility(
    model_name,
    expected_features,
    tmp_path,
    monkeypatch,
):
    # Create an isolated validation split.
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    (splits_dir / "validation.csv").write_text(
        f"path,action\nsample.avi,{ACTION_CLASSES[0]}\n",
        encoding="utf-8",
    )

    # Redirect dataset paths to temporary directories.
    monkeypatch.setattr(
        "src.data.pose_dataset.SPLITS_DIR",
        splits_dir,
    )

    monkeypatch.setattr(
        "src.data.pose_dataset.RAW_DATA_DIR",
        tmp_path,
    )

    # Simulate a cached pose sequence.
    fake_pose = {
        "landmarks": np.zeros((16, 33, 4), dtype=np.float32),
        "detection_mask": np.ones(16, dtype=np.float32),
    }

    monkeypatch.setattr(
        "src.data.pose_dataset.load_cached_pose",
        lambda *args, **kwargs: fake_pose,
    )

    model = build_model(model_name)
    dataset = build_dataset(model_name, "validation")

    assert len(dataset) == 1

    features, label = dataset[0]

    assert features.shape == (16, expected_features)
    assert features.dtype == torch.float32
    assert label == 0

    model.eval()

    with torch.no_grad():
        output = model(features.unsqueeze(0))

    assert output.shape == (1, 6)
    assert torch.isfinite(output).all()