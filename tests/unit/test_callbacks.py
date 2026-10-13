import pytest
import torch

from src.training.callbacks import (
    CheckpointManager,
    EarlyStopping,
)


def test_early_stopping_improvement():
    stopper = EarlyStopping(patience=3)

    assert stopper.step(1.0) is True
    assert stopper.step(0.8) is True

    assert stopper.best_loss == pytest.approx(0.8)
    assert stopper.counter == 0
    assert stopper.should_stop is False


def test_early_stopping_patience():
    stopper = EarlyStopping(patience=2)

    stopper.step(0.5)

    assert stopper.step(0.6) is False
    assert stopper.should_stop is False

    assert stopper.step(0.7) is False
    assert stopper.should_stop is True


def test_early_stopping_counter_reset():
    stopper = EarlyStopping(patience=3)

    stopper.step(1.0)
    stopper.step(1.1)

    assert stopper.counter == 1

    assert stopper.step(0.9) is True
    assert stopper.counter == 0


def test_early_stopping_state_restoration():
    original = EarlyStopping(patience=3)

    original.step(1.0)
    original.step(1.1)

    restored = EarlyStopping(patience=3)
    restored.load_state_dict(original.state_dict())

    assert restored.best_loss == original.best_loss
    assert restored.counter == original.counter
    assert restored.should_stop == original.should_stop


@pytest.mark.parametrize("patience", [0, -1])
def test_early_stopping_invalid_patience(patience):
    with pytest.raises(ValueError):
        EarlyStopping(patience=patience)


def test_checkpoint_save_and_load(tmp_path):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    manager = CheckpointManager(tmp_path / "checkpoints")
    stopper = EarlyStopping(patience=3)
    stopper.step(0.5)

    original_weights = model.weight.detach().clone()

    manager.save(
        epoch=4,
        model=model,
        optimizer=optimizer,
        validation_loss=0.5,
        best_validation_loss=0.5,
        is_best=True,
        early_stopping=stopper,
    )

    assert manager.best_path.exists()
    assert manager.last_path.exists()

    with torch.no_grad():
        model.weight.zero_()

    checkpoint = manager.load(
        model=model,
        optimizer=optimizer,
        best=True,
        early_stopping=stopper,
    )

    assert checkpoint["epoch"] == 4
    assert checkpoint["validation_loss"] == pytest.approx(0.5)
    assert torch.equal(model.weight, original_weights)


def test_checkpoint_last_and_best_are_separate(tmp_path):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    manager = CheckpointManager(tmp_path)

    manager.save(
        epoch=1,
        model=model,
        optimizer=optimizer,
        validation_loss=0.5,
        best_validation_loss=0.5,
        is_best=True,
    )

    manager.save(
        epoch=2,
        model=model,
        optimizer=optimizer,
        validation_loss=0.7,
        best_validation_loss=0.5,
        is_best=False,
    )

    best_checkpoint = manager.load(model, best=True)
    last_checkpoint = manager.load(model, best=False)

    assert best_checkpoint["epoch"] == 1
    assert last_checkpoint["epoch"] == 2


def test_checkpoint_missing_file(tmp_path):
    model = torch.nn.Linear(4, 2)

    manager = CheckpointManager(tmp_path)

    with pytest.raises(FileNotFoundError):
        manager.load(model)