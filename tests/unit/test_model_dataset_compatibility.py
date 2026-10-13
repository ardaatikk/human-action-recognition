import pytest
import torch

from src.training.train import build_model, build_dataset


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
):
    model = build_model(model_name)
    dataset = build_dataset(model_name, "validation")

    features, label = dataset[0]

    assert features.shape == (16, expected_features)

    model.eval()

    with torch.no_grad():
        output = model(features.unsqueeze(0))

    assert output.shape == (1, 6)
    assert torch.isfinite(output).all()