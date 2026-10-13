import pandas as pd
import pytest

from src.analysis.pose_detection_impact import (
    summarize_by_detection_group,
    summarize_by_class,
    summarize_class_and_group,
)


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "action": [
                "boxing",
                "boxing",
                "jogging",
                "jogging",
                "running",
                "running",
            ],
            "correct": [
                True,
                False,
                True,
                False,
                True,
                True,
            ],
            "detection_rate": [
                1.0,
                0.9,
                0.2,
                0.3,
                0.5,
                0.6,
            ],
            "detection_group": pd.Categorical(
                [
                    "High",
                    "High",
                    "Low",
                    "Low",
                    "Medium",
                    "Medium",
                ],
                categories=["Low", "Medium", "High"],
                ordered=True,
            ),
        }
    )


def test_detection_group_summary(sample_dataframe):
    summary = summarize_by_detection_group(sample_dataframe)

    low = summary.loc[
        summary["detection_group"] == "Low"
    ].iloc[0]

    assert low["videos"] == 2
    assert low["correct"] == 1
    assert low["accuracy"] == pytest.approx(0.5)


def test_detection_group_summary_high(sample_dataframe):
    summary = summarize_by_detection_group(sample_dataframe)

    high = summary.loc[
        summary["detection_group"] == "High"
    ].iloc[0]

    assert high["videos"] == 2
    assert high["accuracy"] == pytest.approx(0.5)


def test_class_summary(sample_dataframe):
    summary = summarize_by_class(sample_dataframe)

    running = summary.loc[
        summary["action"] == "running"
    ].iloc[0]

    assert running["videos"] == 2
    assert running["correct"] == 2
    assert running["accuracy"] == pytest.approx(1.0)


def test_class_detection_summary(sample_dataframe):
    summary = summarize_class_and_group(sample_dataframe)

    jogging_low = summary.loc[
        (summary["action"] == "jogging")
        & (summary["detection_group"] == "Low")
    ].iloc[0]

    assert jogging_low["videos"] == 2
    assert jogging_low["accuracy"] == pytest.approx(0.5)