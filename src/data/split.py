from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.paths import REPORTS_DIR, SPLITS_DIR
from src.config.settings import ACTION_CLASSES, RANDOM_SEED

EXCLUDED_VIDEOS = {
    "person01_boxing_d4_uncomp.avi",
}

TRAIN_SUBJECTS = 15
VALIDATION_SUBJECTS = 5
TEST_SUBJECTS = 5


def create_subject_splits(
    dataframe: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """Create reproducible subject-independent dataset splits."""

    if dataframe.empty:
        raise ValueError("Dataset metadata is empty.")

    if dataframe["person_id"].isna().any():
        raise ValueError("Dataset contains missing person IDs.")

    if not dataframe["readable"].all():
        raise ValueError("Dataset contains unreadable videos.")

    subjects = np.array(
        sorted(dataframe["person_id"].unique())
    )

    expected_subjects = (
        TRAIN_SUBJECTS
        + VALIDATION_SUBJECTS
        + TEST_SUBJECTS
    )

    if len(subjects) != expected_subjects:
        raise ValueError(
            f"Expected {expected_subjects} subjects, "
            f"found {len(subjects)}."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    train_subjects = subjects[:TRAIN_SUBJECTS]

    validation_subjects = subjects[
        TRAIN_SUBJECTS:
        TRAIN_SUBJECTS + VALIDATION_SUBJECTS
    ]

    test_subjects = subjects[
        TRAIN_SUBJECTS + VALIDATION_SUBJECTS:
    ]

    subject_groups = {
        "train": train_subjects,
        "validation": validation_subjects,
        "test": test_subjects,
    }

    splits = {}

    for split_name, subject_ids in subject_groups.items():
        split_dataframe = dataframe[
            dataframe["person_id"].isin(subject_ids)
        ].copy()

        split_dataframe = split_dataframe.sort_values(
            ["person_id", "action", "scenario"]
        ).reset_index(drop=True)

        splits[split_name] = split_dataframe

    return splits


def validate_splits(
    splits: dict[str, pd.DataFrame],
) -> None:
    """Ensure subject independence and complete video coverage."""

    subject_sets = {
        name: set(dataframe["person_id"])
        for name, dataframe in splits.items()
    }

    if subject_sets["train"] & subject_sets["validation"]:
        raise ValueError("Train/validation subject leakage.")

    if subject_sets["train"] & subject_sets["test"]:
        raise ValueError("Train/test subject leakage.")

    if subject_sets["validation"] & subject_sets["test"]:
        raise ValueError("Validation/test subject leakage.")

    filenames = []

    for dataframe in splits.values():
        filenames.extend(dataframe["filename"].tolist())

    if len(filenames) != len(set(filenames)):
        raise ValueError("Duplicate videos across splits.")

    print("[OK] No subject leakage detected.")
    print("[OK] No duplicate videos across splits.")


def save_splits(
    splits: dict[str, pd.DataFrame],
) -> None:
    """Save split metadata as CSV files."""

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, dataframe in splits.items():
        output_path = SPLITS_DIR / f"{split_name}.csv"

        dataframe.to_csv(output_path, index=False)

        print(f"[SAVED] {output_path}")


def print_split_summary(
    splits: dict[str, pd.DataFrame],
) -> None:
    """Print subject and class distributions."""

    print("\n" + "=" * 60)
    print("SUBJECT-INDEPENDENT DATASET SPLITS")
    print("=" * 60)

    for split_name, dataframe in splits.items():
        subjects = sorted(
            dataframe["person_id"].unique().tolist()
        )

        print(f"\n{split_name.upper()}")
        print(f"Subjects ({len(subjects)}): {subjects}")
        print(f"Videos: {len(dataframe)}")

        class_counts = (
            dataframe["action"]
            .value_counts()
            .reindex(ACTION_CLASSES, fill_value=0)
        )

        print(class_counts.to_string())


def main() -> None:
    metadata_path = REPORTS_DIR / "dataset_validation.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            "Dataset validation report not found. "
            "Run python -m src.data.validate first."
        )

    dataframe = pd.read_csv(metadata_path)
    
    # Exclude videos with confirmed decoding corruption.
    excluded_mask = dataframe["filename"].isin(EXCLUDED_VIDEOS)

    excluded_count = int(excluded_mask.sum())

    dataframe = dataframe.loc[~excluded_mask].copy()

    print(f"[INFO] Excluded corrupted videos: {excluded_count}")

    splits = create_subject_splits(dataframe)

    validate_splits(splits)

    save_splits(splits)

    print_split_summary(splits)


if __name__ == "__main__":
    main()