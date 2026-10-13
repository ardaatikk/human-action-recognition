from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from src.config.paths import RAW_DATA_DIR, SPLITS_DIR
from src.config.settings import ACTION_CLASSES
from src.data.preprocess import load_video_frames
from src.data.cache import load_cached_frames


class KTHVideoDataset(Dataset):
    """PyTorch Dataset for KTH Human Actions videos."""

    def __init__(self, split: str, use_cache: bool = True) -> None:
        
        if split not in {"train", "validation", "test"}:
            raise ValueError(
                f"Invalid dataset split: {split}"
            )

        split_path = SPLITS_DIR / f"{split}.csv"

        if not split_path.exists():
            raise FileNotFoundError(split_path)

        self.dataframe = pd.read_csv(split_path)

        self.class_to_idx = {
            action: index
            for index, action in enumerate(ACTION_CLASSES)
        }

        unknown_actions = (
            set(self.dataframe["action"])
            - set(self.class_to_idx)
        )

        if unknown_actions:
            raise ValueError(
                f"Unknown actions: {unknown_actions}"
            )
            
        self.use_cache = use_cache

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        row = self.dataframe.iloc[index]

        video_path = RAW_DATA_DIR / row["path"]

        if self.use_cache:
            frames = load_cached_frames(video_path)
        else:
            frames = load_video_frames(video_path)

        # (T, H, W) -> (T, C, H, W)
        frames = np.expand_dims(frames, axis=1)

        video_tensor = torch.from_numpy(frames)

        label = self.class_to_idx[row["action"]]

        label_tensor = torch.tensor(
            label,
            dtype=torch.long,
        )

        return video_tensor, label_tensor