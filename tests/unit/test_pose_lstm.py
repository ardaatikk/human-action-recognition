import pytest
import torch
from torch import nn

from src.models.pose_lstm import PoseLSTM


def test_default_output_shape():
    model = PoseLSTM()

    x = torch.randn(8, 16, 135)

    output = model(x)

    assert output.shape == (8, 6)


def test_single_sample():
    model = PoseLSTM()

    x = torch.randn(1, 16, 135)

    output = model(x)

    assert output.shape == (1, 6)


def test_variable_sequence_length():
    model = PoseLSTM()

    x = torch.randn(4, 32, 135)

    output = model(x)

    assert output.shape == (4, 6)


def test_custom_configuration():
    model = PoseLSTM(
        input_size=64,
        hidden_size=32,
        num_layers=1,
        num_classes=3,
        dropout=0.2,
    )

    x = torch.randn(2, 10, 64)

    output = model(x)

    assert output.shape == (2, 3)


def test_output_contains_finite_values():
    model = PoseLSTM()

    x = torch.randn(4, 16, 135)

    output = model(x)

    assert torch.isfinite(output).all()


def test_backward_pass():
    model = PoseLSTM()

    x = torch.randn(4, 16, 135)

    targets = torch.tensor([0, 1, 2, 3])

    criterion = nn.CrossEntropyLoss()

    logits = model(x)

    loss = criterion(logits, targets)

    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_optimizer_updates_parameters():
    torch.manual_seed(42)

    model = PoseLSTM()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    criterion = nn.CrossEntropyLoss()

    x = torch.randn(4, 16, 135)
    targets = torch.tensor([0, 1, 2, 3])

    original_weights = (
        model.classifier.weight.detach().clone()
    )

    optimizer.zero_grad()

    logits = model(x)

    loss = criterion(logits, targets)

    loss.backward()

    optimizer.step()

    updated_weights = model.classifier.weight.detach()

    assert not torch.equal(
        original_weights,
        updated_weights,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_size": 0},
        {"hidden_size": 0},
        {"num_layers": 0},
        {"num_classes": 0},
        {"dropout": -0.1},
        {"dropout": 1.0},
    ],
)
def test_invalid_model_configuration(kwargs):
    with pytest.raises(ValueError):
        PoseLSTM(**kwargs)


def test_invalid_input_dimensions():
    model = PoseLSTM()

    x = torch.randn(16, 135)

    with pytest.raises(ValueError):
        model(x)


def test_invalid_feature_dimension():
    model = PoseLSTM()

    x = torch.randn(4, 16, 100)

    with pytest.raises(ValueError):
        model(x)


def test_empty_sequence():
    model = PoseLSTM()

    x = torch.empty(4, 0, 135)

    with pytest.raises(ValueError):
        model(x)


def test_eval_mode_is_deterministic():
    model = PoseLSTM()
    model.eval()

    x = torch.randn(4, 16, 135)

    with torch.no_grad():
        output_1 = model(x)
        output_2 = model(x)

    torch.testing.assert_close(
        output_1,
        output_2,
    )


def test_parameter_count():
    model = PoseLSTM()

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert total_parameters == 268_550