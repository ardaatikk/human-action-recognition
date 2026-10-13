from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config.paths import RAW_DATA_DIR, SPLITS_DIR
from src.config.settings import ACTION_CLASSES, NUM_FRAMES
from src.features.pose_cache import load_cached_pose
from src.features.pose_normalization import normalize_pose_sequence


class KTHPoseDataset(Dataset):
    """
    PyTorch Dataset for cached KTH pose sequences.

    Each sample:
        features: (T, 135), float32
        label: integer class index
    """

    def __init__(
        self,
        split: str,
        num_frames: int = NUM_FRAMES,
    ) -> None:
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Invalid split: {split}")

        if num_frames <= 0:
            raise ValueError("num_frames must be positive")

        self.split = split
        self.num_frames = num_frames

        split_path = SPLITS_DIR / f"{split}.csv"

        if not split_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_path}"
            )

        self.data = pd.read_csv(split_path)

        required_columns = {"path", "action"}

        missing_columns = required_columns - set(self.data.columns)

        if missing_columns:
            raise ValueError(
                f"Missing CSV columns: {sorted(missing_columns)}"
            )

        self.class_to_idx = {
            action: index
            for index, action in enumerate(ACTION_CLASSES)
        }

        unknown_actions = (
            set(self.data["action"].dropna().unique())
            - set(self.class_to_idx)
        )

        if unknown_actions or self.data["action"].isna().any():
            raise ValueError(
                f"Invalid action labels: {unknown_actions}"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int]:

        row = self.data.iloc[index]

        video_path = RAW_DATA_DIR / row["path"]

        pose_data = load_cached_pose(
            video_path,
            num_frames=self.num_frames,
        )

        features = normalize_pose_sequence(
            landmarks=pose_data["landmarks"],
            detection_mask=pose_data["detection_mask"],
        )

        label = self.class_to_idx[row["action"]]

        features_tensor = torch.from_numpy(features)

        return features_tensor, label