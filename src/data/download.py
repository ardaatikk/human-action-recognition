from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from src.config.paths import RAW_DATA_DIR
from src.config.settings import ACTION_CLASSES


BASE_URL = "https://www.csc.kth.se/cvap/actions"

ARCHIVE_DIR = RAW_DATA_DIR / "archives"
VIDEO_DIR = RAW_DATA_DIR / "videos"

CHUNK_SIZE = 1024 * 1024


def download_file(url: str, destination: Path) -> None:
    """Download a file with a progress indicator."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = destination.with_name(
        destination.name + ".part"
    )

    if destination.exists():
        print(f"[SKIP] Already downloaded: {destination.name}")
        return

    print(f"[DOWNLOAD] {destination.name}")

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total_size = int(
                response.headers.get("Content-Length", 0)
            )

            downloaded = 0

            with temporary_file.open("wb") as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)

                    if not chunk:
                        break

                    output.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        percentage = downloaded / total_size * 100

                        print(
                            f"\rProgress: {percentage:6.2f}%",
                            end="",
                            flush=True,
                        )

        if total_size and downloaded != total_size:
            raise IOError(
                f"Incomplete download: {downloaded}/{total_size} bytes"
            )

        temporary_file.replace(destination)

        print(f"\n[OK] Downloaded: {destination.name}")

    except Exception:
        temporary_file.unlink(missing_ok=True)
        raise


def validate_archive(archive_path: Path) -> bool:
    """Check whether a ZIP archive is readable and complete."""

    if not zipfile.is_zipfile(archive_path):
        return False

    try:
        with zipfile.ZipFile(archive_path) as archive:
            return archive.testzip() is None

    except (OSError, zipfile.BadZipFile):
        return False


def extract_archive(
    archive_path: Path,
    destination: Path,
) -> None:
    """Extract a ZIP archive safely."""

    destination.mkdir(parents=True, exist_ok=True)

    marker = destination / ".extraction_complete"

    if marker.exists():
        print(f"[SKIP] Already extracted: {archive_path.name}")
        return

    print(f"[EXTRACT] {archive_path.name}")

    with zipfile.ZipFile(archive_path) as archive:
        root = destination.resolve()

        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()

            if not member_path.is_relative_to(root):
                raise ValueError(
                    f"Unsafe archive entry: {member.filename}"
                )

            if member.is_dir():
                continue

            if not member.filename.lower().endswith(".avi"):
                continue

            output_path = destination / Path(member.filename).name

            with archive.open(member) as source:
                with output_path.open("wb") as target:
                    shutil.copyfileobj(source, target)

    marker.touch()

    print(f"[OK] Extracted: {archive_path.name}")


def download_dataset(
    keep_archives: bool = False,
) -> None:
    """Download and extract the KTH Human Actions dataset."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("KTH HUMAN ACTIONS DATASET")
    print("=" * 55)

    metadata_url = f"{BASE_URL}/00sequences.txt"
    metadata_path = RAW_DATA_DIR / "00sequences.txt"

    download_file(metadata_url, metadata_path)

    for action in ACTION_CLASSES:
        archive_name = f"{action}.zip"

        archive_url = f"{BASE_URL}/{archive_name}"
        archive_path = ARCHIVE_DIR / archive_name

        destination = VIDEO_DIR / action

        if (
            (destination / ".extraction_complete").exists()
            and not keep_archives
        ):
            print(f"[SKIP] Already extracted: {action}")
            continue

        if archive_path.exists() and not validate_archive(archive_path):
            print(f"[WARNING] Invalid archive: {archive_name}")
            archive_path.unlink()

        download_file(archive_url, archive_path)

        if not validate_archive(archive_path):
            archive_path.unlink(missing_ok=True)

            raise RuntimeError(
                f"Archive validation failed: {archive_name}"
            )

        extract_archive(archive_path, destination)

        if not keep_archives:
            archive_path.unlink(missing_ok=True)

    print("\n" + "=" * 55)
    print("DATASET DOWNLOAD COMPLETED")
    print("=" * 55)

    total_videos = 0

    for action in ACTION_CLASSES:
        action_dir = VIDEO_DIR / action

        count = len(list(action_dir.glob("*.avi")))

        total_videos += count

        print(f"{action:15s}: {count:4d} videos")

    print("-" * 55)
    print(f"Total videos: {total_videos}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the KTH Human Actions dataset."
    )

    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded ZIP archives after extraction.",
    )

    args = parser.parse_args()

    download_dataset(
        keep_archives=args.keep_archives,
    )


if __name__ == "__main__":
    main()