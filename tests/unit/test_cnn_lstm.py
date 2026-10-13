import pytest
import torch

from src.models.cnn_lstm import (
    CNNFeatureExtractor,
    CNNLSTM,
)


def test_cnn_feature_extractor_output_shape():
    model = CNNFeatureExtractor(feature_dim=128)

    x = torch.randn(4, 1, 112, 112)

    output = model(x)

    assert output.shape == (4, 128)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_cnn_lstm_output_shape(batch_size):
    model = CNNLSTM()

    x = torch.randn(
        batch_size,
        16,
        1,
        112,
        112,
    )

    output = model(x)

    assert output.shape == (batch_size, 6)


def test_cnn_lstm_gradient_flow():
    model = CNNLSTM()

    x = torch.randn(2, 16, 1, 112, 112)

    targets = torch.tensor([0, 1])

    criterion = torch.nn.CrossEntropyLoss()

    outputs = model(x)

    loss = criterion(outputs, targets)

    loss.backward()

    assert model.cnn.features[0].weight.grad is not None
    assert model.lstm.weight_ih_l0.grad is not None
    assert model.classifier[-1].weight.grad is not None


def test_cnn_lstm_output_is_finite():
    model = CNNLSTM()

    x = torch.randn(2, 16, 1, 112, 112)

    model.eval()

    with torch.no_grad():
        output = model(x)

    assert torch.isfinite(output).all()


def test_cnn_lstm_parameter_count():
    model = CNNLSTM()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert parameter_count > 0
    assert parameter_count < 1_000_000