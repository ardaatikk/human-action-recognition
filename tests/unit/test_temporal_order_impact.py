import torch
import pytest

from src.analysis.temporal_order_impact import reorder_sequence


@pytest.fixture
def sample_sequence():
    return torch.arange(
        24,
        dtype=torch.float32,
    ).reshape(2, 4, 3)


def test_original_order(sample_sequence):
    result = reorder_sequence(
        sample_sequence,
        mode="original",
    )

    assert torch.equal(result, sample_sequence)


def test_reversed_order(sample_sequence):
    result = reorder_sequence(
        sample_sequence,
        mode="reversed",
    )

    expected = torch.flip(
        sample_sequence,
        dims=[1],
    )

    assert torch.equal(result, expected)


def test_shuffled_preserves_frames(sample_sequence):
    generator = torch.Generator().manual_seed(42)

    result = reorder_sequence(
        sample_sequence,
        mode="shuffled",
        generator=generator,
    )

    assert result.shape == sample_sequence.shape

    for batch_index in range(sample_sequence.shape[0]):
        original_frames = {
            tuple(frame.tolist())
            for frame in sample_sequence[batch_index]
        }

        shuffled_frames = {
            tuple(frame.tolist())
            for frame in result[batch_index]
        }

        assert original_frames == shuffled_frames


def test_shuffled_is_reproducible(sample_sequence):
    first = reorder_sequence(
        sample_sequence,
        mode="shuffled",
        generator=torch.Generator().manual_seed(42),
    )

    second = reorder_sequence(
        sample_sequence,
        mode="shuffled",
        generator=torch.Generator().manual_seed(42),
    )

    assert torch.equal(first, second)


def test_shuffled_requires_generator(sample_sequence):
    with pytest.raises(ValueError):
        reorder_sequence(
            sample_sequence,
            mode="shuffled",
        )


def test_invalid_mode(sample_sequence):
    with pytest.raises(ValueError):
        reorder_sequence(
            sample_sequence,
            mode="invalid",
        )