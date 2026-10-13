import numpy as np
import pytest

from src.config.settings import FRAME_HEIGHT, FRAME_WIDTH
from src.data.preprocess import (
    preprocess_frame,
    sample_frame_indices,
)


def test_sample_frame_indices_shape():
    indices = sample_frame_indices(
        total_frames=100,
        num_frames=16,
    )

    assert indices.shape == (16,)
    assert indices[0] == 0
    assert indices[-1] == 99


def test_sample_frame_indices_order():
    indices = sample_frame_indices(
        total_frames=100,
        num_frames=16,
    )

    assert np.all(np.diff(indices) >= 0)


def test_sample_frame_indices_short_video():
    indices = sample_frame_indices(
        total_frames=5,
        num_frames=16,
    )

    assert len(indices) == 16
    assert indices.min() >= 0
    assert indices.max() < 5


@pytest.mark.parametrize(
    "total_frames,num_frames",
    [
        (0, 16),
        (-1, 16),
        (100, 0),
        (100, -1),
    ],
)
def test_sample_frame_indices_invalid_inputs(
    total_frames,
    num_frames,
):
    with pytest.raises(ValueError):
        sample_frame_indices(
            total_frames=total_frames,
            num_frames=num_frames,
        )


def test_preprocess_frame_shape():
    frame = np.zeros(
        (120, 160, 3),
        dtype=np.uint8,
    )

    processed = preprocess_frame(frame)

    assert processed.shape == (
        FRAME_HEIGHT,
        FRAME_WIDTH,
    )


def test_preprocess_frame_normalization():
    frame = np.full(
        (120, 160, 3),
        255,
        dtype=np.uint8,
    )

    processed = preprocess_frame(frame)

    assert processed.dtype == np.float32
    assert processed.min() == pytest.approx(1.0)
    assert processed.max() == pytest.approx(1.0)


def test_preprocess_frame_black_image():
    frame = np.zeros(
        (120, 160, 3),
        dtype=np.uint8,
    )

    processed = preprocess_frame(frame)

    assert processed.min() == pytest.approx(0.0)
    assert processed.max() == pytest.approx(0.0)


def test_preprocess_frame_grayscale():
    frame = np.zeros(
        (120, 160),
        dtype=np.uint8,
    )

    processed = preprocess_frame(frame)

    assert processed.shape == (
        FRAME_HEIGHT,
        FRAME_WIDTH,
    )


def test_preprocess_frame_invalid_dimensions():
    frame = np.zeros(
        (120,),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        preprocess_frame(frame)