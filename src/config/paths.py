from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Artifacts
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODELS_DIR = ARTIFACTS_DIR / "models"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# Training outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TRAINING_OUTPUTS_DIR = OUTPUTS_DIR / "training"


def ensure_directories() -> None:
    """Create required project directories if they do not exist."""

    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SPLITS_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        TRAINING_OUTPUTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)