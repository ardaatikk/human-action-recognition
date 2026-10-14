"""Unit tests for multi-seed experiment summarization."""

import json

import pytest

from src.analysis import summarize_experiments


def test_summarize_calculates_mean_and_sample_std():
    mean, std = summarize_experiments.summarize(
        [0.80, 0.85, 0.90]
    )

    assert mean == pytest.approx(0.85)
    assert std == pytest.approx(0.05)


def test_summarize_requires_multiple_runs():
    with pytest.raises(ValueError, match="At least two runs"):
        summarize_experiments.summarize([0.85])


def test_load_metrics_reads_expected_values(tmp_path, monkeypatch):
    experiment = "test_experiment"

    metrics_dir = (
        tmp_path
        / experiment
        / "validation"
        / "best"
    )
    metrics_dir.mkdir(parents=True)

    metrics_path = metrics_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps({
            "accuracy": 0.85,
            "f1_macro": 0.82,
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        summarize_experiments,
        "REPORTS_DIR",
        tmp_path,
    )

    metrics = summarize_experiments.load_metrics(experiment)

    assert metrics["accuracy"] == pytest.approx(0.85)
    assert metrics["f1_macro"] == pytest.approx(0.82)


def test_load_metrics_raises_for_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        summarize_experiments,
        "REPORTS_DIR",
        tmp_path,
    )

    with pytest.raises(FileNotFoundError):
        summarize_experiments.load_metrics("missing_experiment")


def test_load_metrics_raises_for_missing_metric(tmp_path, monkeypatch):
    metrics_dir = (
        tmp_path
        / "incomplete_experiment"
        / "validation"
        / "best"
    )
    metrics_dir.mkdir(parents=True)

    (metrics_dir / "metrics.json").write_text(
        json.dumps({"accuracy": 0.85}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        summarize_experiments,
        "REPORTS_DIR",
        tmp_path,
    )

    with pytest.raises(KeyError, match="f1_macro"):
        summarize_experiments.load_metrics(
            "incomplete_experiment"
        )