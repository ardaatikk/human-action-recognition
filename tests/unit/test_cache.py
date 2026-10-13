from pathlib import Path

import numpy as np
import pytest

from src.data import cache


def test_cache_directory_contains_preprocessing_parameters():
    directory = cache.get_cache_dir(
        num_frames=16,
        height=112,
        width=112,
    )

    assert directory.name == "v1_t16_h112_w112"


def test_cache_path_preserves_video_structure(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    monkeypatch.setattr(cache, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(cache, "PROCESSED_DATA_DIR", processed_dir)

    video = (
        raw_dir
        / "videos"
        / "boxing"
        / "person02_boxing_d1_uncomp.avi"
    )

    expected = (
        processed_dir
        / "frames"
        / "v1_t16_h112_w112"
        / "videos"
        / "boxing"
        / "person02_boxing_d1_uncomp.npy"
    )

    assert cache.get_cache_path(video) == expected


def test_cache_creates_file(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    monkeypatch.setattr(cache, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(cache, "PROCESSED_DATA_DIR", processed_dir)

    video = raw_dir / "videos" / "boxing" / "sample.avi"

    expected_frames = np.full(
        (16, 112, 112),
        0.5,
        dtype=np.float32,
    )

    def fake_load_video_frames(video_path, num_frames):
        assert video_path == video
        assert num_frames == 16
        return expected_frames

    monkeypatch.setattr(
        cache,
        "load_video_frames",
        fake_load_video_frames,
    )

    frames = cache.load_cached_frames(video)

    assert np.array_equal(frames, expected_frames)
    assert cache.get_cache_path(video).exists()


def test_cache_reuses_existing_file(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    monkeypatch.setattr(cache, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(cache, "PROCESSED_DATA_DIR", processed_dir)

    video = raw_dir / "videos" / "walking" / "sample.avi"

    cached_frames = np.full(
        (16, 112, 112),
        0.75,
        dtype=np.float32,
    )

    cache_path = cache.get_cache_path(video)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(cache_path, cached_frames)

    def should_not_decode(*args, **kwargs):
        raise AssertionError(
            "Video should not be decoded when cache exists."
        )

    monkeypatch.setattr(
        cache,
        "load_video_frames",
        should_not_decode,
    )

    frames = cache.load_cached_frames(video)

    assert np.array_equal(frames, cached_frames)


def test_cache_rejects_unexpected_shape(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    monkeypatch.setattr(cache, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(cache, "PROCESSED_DATA_DIR", processed_dir)

    video = raw_dir / "videos" / "running" / "sample.avi"

    def fake_load_video_frames(video_path, num_frames):
        return np.zeros(
            (8, 112, 112),
            dtype=np.float32,
        )

    monkeypatch.setattr(
        cache,
        "load_video_frames",
        fake_load_video_frames,
    )

    with pytest.raises(ValueError, match="Unexpected shape"):
        cache.load_cached_frames(video)

    assert not cache.get_cache_path(video).exists()


def test_cache_path_rejects_video_outside_raw_directory(
    tmp_path,
    monkeypatch,
):
    raw_dir = tmp_path / "raw"

    monkeypatch.setattr(cache, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(
        cache,
        "PROCESSED_DATA_DIR",
        tmp_path / "processed",
    )

    outside_video = tmp_path / "other" / "sample.avi"

    with pytest.raises(ValueError):
        cache.get_cache_path(outside_video)