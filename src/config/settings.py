import torch


# Reproducibility
RANDOM_SEED = 42

# Dataset
DATASET_NAME = "KTH Human Actions"

ACTION_CLASSES = (
    "boxing",
    "handclapping",
    "handwaving",
    "jogging",
    "running",
    "walking",
)

NUM_CLASSES = len(ACTION_CLASSES)

# Video preprocessing
NUM_FRAMES = 16
FRAME_HEIGHT = 112
FRAME_WIDTH = 112

# Training
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30

# Device
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Training configuration
EARLY_STOPPING_PATIENCE = 7
EARLY_STOPPING_MIN_DELTA = 0.0

NUM_WORKERS = 0
WEIGHT_DECAY = 1e-4

DEFAULT_EXPERIMENT_NAME = "cnn_lstm_baseline"