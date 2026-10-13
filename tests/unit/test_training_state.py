import random

import numpy as np
import torch

from src.training.train import (
    capture_rng_state,
    restore_rng_state,
)
from src.utils.seed import set_seed


def test_rng_state_restoration():
    set_seed(42)

    saved_state = capture_rng_state()

    expected_python = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(5)

    restore_rng_state(saved_state)

    assert random.random() == expected_python
    assert np.random.rand() == expected_numpy
    assert torch.equal(torch.rand(5), expected_torch)


def test_rng_state_restoration_after_checkpoint(tmp_path):
    set_seed(42)

    saved_state = capture_rng_state()

    checkpoint_path = tmp_path / "rng_checkpoint.pt"

    torch.save(
        {"rng_state": saved_state},
        checkpoint_path,
    )

    expected = torch.rand(5)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    restore_rng_state(checkpoint["rng_state"])

    actual = torch.rand(5)

    assert torch.equal(actual, expected)


def test_dataloader_generator_state_restoration():
    generator = torch.Generator()
    generator.manual_seed(42)

    saved_state = generator.get_state()

    expected = torch.randperm(
        10,
        generator=generator,
    )

    generator.set_state(saved_state.cpu().to(torch.uint8))

    actual = torch.randperm(
        10,
        generator=generator,
    )

    assert torch.equal(actual, expected)