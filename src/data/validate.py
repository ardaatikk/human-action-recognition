from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd

from src.config.paths import RAW_DATA_DIR, REPORTS_DIR
from src.config.settings import ACTION_CLASSES


VIDEO_DIR = RAW_DATA_DIR / "videos"

FILENAME_PATTERN = re.compile(
    r"^person(?P<person>\d{2})_"
    r"(?P<action>[a-z]+)_"
    r"d(?P<scenario>\d+)_"
    r"uncomp\.avi$"
)


def inspect_video(video_path: Path) -> dict:
    """Inspect video metadata and readability."""

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        return {
            "readable": False,
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "frame_count": 0,
        }

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    first_frame_ok, _ = capture.read()

    capture.release()

    return {
        "readable": bool(
            first_frame_ok
            and width > 0
            and height > 0
            and fps > 0
            and frame_count > 0
        ),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
    }


def validate_dataset() -> pd.DataFrame:
    """Validate KTH videos and generate a metadata report."""

    records = []

    for action in ACTION_CLASSES:
        action_dir = VIDEO_DIR / action

        if not action_dir.exists():
            raise FileNotFoundError(
                f"Missing action directory: {action_dir}"
            )

        video_paths = sorted(action_dir.glob("*.avi"))

        print(f"[CHECK] {action}: {len(video_paths)} videos")

        for video_path in video_paths:
            match = FILENAME_PATTERN.match(video_path.name)

            if match:
                person_id = int(match.group("person"))
                scenario = int(match.group("scenario"))
                filename_action = match.group("action")
            else:
                person_id = None
                scenario = None
                filename_action = None

            metadata = inspect_video(video_path)

            records.append(
                {
                    "filename": video_path.name,
                    "action": action,
                    "filename_action": filename_action,
                    "person_id": person_id,
                    "scenario": scenario,
                    "path": str(video_path.relative_to(RAW_DATA_DIR)),
                    **metadata,
                }
            )

    dataframe = pd.DataFrame(records)

    if dataframe.empty:
        raise RuntimeError("No videos found.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORTS_DIR / "dataset_validation.csv"

    dataframe.to_csv(report_path, index=False)

    print("\n" + "=" * 55)
    print("DATASET VALIDATION REPORT")
    print("=" * 55)

    print(f"Total videos: {len(dataframe)}")
    print(f"Readable videos: {dataframe['readable'].sum()}")
    print(f"Unreadable videos: {(~dataframe['readable']).sum()}")

    print(
        "Unique persons:",
        dataframe["person_id"].nunique(),
    )

    print(
        "Unrecognized filenames:",
        dataframe["person_id"].isna().sum(),
    )

    print(
        "Action mismatches:",
        (
            dataframe["action"]
            != dataframe["filename_action"]
        ).sum(),
    )

    print(
        "Duplicate filenames:",
        dataframe["filename"].duplicated().sum(),
    )

    print("\nResolution distribution:")

    print(
        dataframe.groupby(
            ["width", "height"]
        ).size()
    )

    print("\nFPS distribution:")

    print(
        dataframe["fps"].value_counts().sort_index()
    )

    print("\nPerson distribution:")

    print(
        dataframe.groupby("person_id")
        .size()
        .to_string()
    )

    print(f"\nReport saved: {report_path}")

    return dataframe


def main() -> None:
    validate_dataset()

def validate_video_decoding() -> pd.DataFrame:
    """Check actual frame decoding across all dataset videos."""

    records = []

    for action in ACTION_CLASSES:
        video_paths = sorted(
            (VIDEO_DIR / action).glob("*.avi")
        )

        print(f"\n[DECODING] {action}")

        for video_path in video_paths:
            capture = cv2.VideoCapture(str(video_path))

            expected_frames = int(
                capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            decoded_frames = 0

            if capture.isOpened():
                while True:
                    success, frame = capture.read()

                    if not success:
                        break

                    decoded_frames += 1

            capture.release()

            difference = expected_frames - decoded_frames

            records.append(
                {
                    "filename": video_path.name,
                    "action": action,
                    "expected_frames": expected_frames,
                    "decoded_frames": decoded_frames,
                    "difference": difference,
                    "complete": (
                        expected_frames > 0
                        and decoded_frames == expected_frames
                    ),
                }
            )

    dataframe = pd.DataFrame(records)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / "decoding_validation.csv"

    dataframe.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("VIDEO DECODING VALIDATION")
    print("=" * 60)

    print(f"Total videos: {len(dataframe)}")
    print(f"Complete videos: {dataframe['complete'].sum()}")
    print(f"Incomplete videos: {(~dataframe['complete']).sum()}")

    incomplete = dataframe[~dataframe["complete"]]

    if not incomplete.empty:
        print("\nVideos requiring attention:")
        print(incomplete.to_string(index=False))

    print(f"\nReport saved: {output_path}")

    return dataframe

if __name__ == "__main__":
    main()