from __future__ import annotations

import os
import random

import numpy as np
import torch

from src.config.settings import RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducible experiments."""

    if seed < 0:
        raise ValueError("Seed must be non-negative.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)