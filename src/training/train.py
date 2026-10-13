from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config.paths import MODELS_DIR, TRAINING_OUTPUTS_DIR
from src.config.settings import (
    BATCH_SIZE,
    DEFAULT_EXPERIMENT_NAME,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_WORKERS,
    RANDOM_SEED,
    WEIGHT_DECAY,
)
from src.data.dataset import KTHVideoDataset
from src.data.pose_dataset import KTHPoseDataset

from src.models.cnn_lstm import CNNLSTM
from src.models.pose_lstm import PoseLSTM
from src.training.callbacks import CheckpointManager, EarlyStopping
from src.training.engine import run_epoch
from src.utils.seed import set_seed


HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "train_accuracy",
    "validation_loss",
    "validation_accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a human action recognition model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        choices=["cnn_lstm", "pose_lstm", "pose_mask_lstm", "pose_coordinates_lstm"],
        default="cnn_lstm",
    )
    parser.add_argument(
        "--experiment",
        default=None,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue training from the last checkpoint.",
    )

    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available.")

    return torch.device(requested)


def build_model(model_name: str) -> nn.Module:
    if model_name == "cnn_lstm":
        return CNNLSTM()

    if model_name == "pose_lstm":
        return PoseLSTM()

    if model_name == "pose_mask_lstm":
        return PoseLSTM(input_size=1)
    
    if model_name == "pose_coordinates_lstm":
        return PoseLSTM(input_size=99)

    raise ValueError(f"Unsupported model: {model_name}")

def build_dataset(
    model_name: str,
    split: str,
) -> KTHVideoDataset | KTHPoseDataset:

    if model_name == "cnn_lstm":
        return KTHVideoDataset(split=split)

    if model_name == "pose_lstm":
        return KTHPoseDataset(
            split=split,
            feature_mode="full",
        )

    if model_name == "pose_mask_lstm":
        return KTHPoseDataset(
            split=split,
            feature_mode="mask_only",
        )
    
    if model_name == "pose_coordinates_lstm":
        return KTHPoseDataset(
            split=split,
            feature_mode="coordinates_only",
        )

    raise ValueError(f"Unsupported model: {model_name}")


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    if args.lr <= 0:
        raise ValueError("--lr must be positive.")

    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")

    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative.")

    if args.seed < 0:
        raise ValueError("--seed cannot be negative.")

    if (
        not args.experiment
        or args.experiment in {".", ".."}
        or Path(args.experiment).name != args.experiment
    ):
        raise ValueError(
            "--experiment must be a single directory name."
        )


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def append_history(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    is_new = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=HISTORY_FIELDS,
        )

        if is_new:
            writer.writeheader()

        writer.writerow(row)


def capture_rng_state() -> dict:
    numpy_state = np.random.get_state()

    state = {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()

    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])

    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state[0],
            np.asarray(numpy_state[1], dtype=np.uint32),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
    )
    
    torch.set_rng_state(
        state["torch"].cpu().to(dtype=torch.uint8)
    )

    if "cuda" in state and torch.cuda.is_available():
        cuda_states = [
            rng_state.cpu().to(dtype=torch.uint8)
            for rng_state in state["cuda"]
        ]

        torch.cuda.set_rng_state_all(cuda_states)

    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(
            state["mps"].cpu().to(dtype=torch.uint8)
        )


def main() -> None:
    args = parse_args()

    if args.experiment is None:
        experiment_names = {
            "cnn_lstm": DEFAULT_EXPERIMENT_NAME,
            "pose_lstm": "pose_lstm_baseline",
            "pose_mask_lstm": "pose_mask_only_ablation",
            "pose_coordinates_lstm": "pose_coordinates_only_ablation",
        }

        args.experiment = experiment_names[args.model]

    validate_args(args)

    set_seed(args.seed)

    device = resolve_device(args.device)

    experiment_dir = TRAINING_OUTPUTS_DIR / args.experiment
    checkpoint_dir = MODELS_DIR / args.experiment

    history_path = experiment_dir / "history.csv"
    config_path = experiment_dir / "config.json"

    if not args.resume and (
        history_path.exists()
        or (checkpoint_dir / "last_checkpoint.pt").exists()
    ):
        raise FileExistsError(
            f"Experiment '{args.experiment}' already exists. "
            "Use --resume or choose a new --experiment name."
        )

    experiment_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": args.model,
        "experiment": args.experiment,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "patience": args.patience,
        "min_delta": EARLY_STOPPING_MIN_DELTA,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
        "num_workers": NUM_WORKERS,
    }

    if args.resume:
        if not config_path.exists():
            raise FileNotFoundError(
                f"Missing experiment configuration: {config_path}"
            )

        with config_path.open(encoding="utf-8") as file:
            previous_config = json.load(file)

        for key in (
            "model",
            "batch_size",
            "learning_rate",
            "patience",
            "min_delta",
            "weight_decay",
            "seed",
            "device",
            "num_workers",
        ):
            if config[key] != previous_config[key]:
                raise ValueError(
                    f"Cannot resume: '{key}' changed. "
                    f"Previous={previous_config[key]}, "
                    f"current={config[key]}"
                )

    print("\n" + "=" * 60)
    print("HUMAN ACTION RECOGNITION — TRAINING")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Experiment:  {args.experiment}")
    print(f"Device:      {device}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)

    train_dataset = build_dataset(
        model_name=args.model,
        split="train",
    )

    validation_dataset = build_dataset(
        model_name=args.model,
        split="validation",
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=train_generator,
        pin_memory=device.type == "cuda",
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    print(f"Training videos:   {len(train_dataset)}")
    print(f"Validation videos: {len(validation_dataset)}")

    model = build_model(args.model).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=EARLY_STOPPING_MIN_DELTA,
    )

    checkpoints = CheckpointManager(checkpoint_dir)

    start_epoch = 1

    if args.resume:
        checkpoint = checkpoints.load(
            model=model,
            optimizer=optimizer,
            device=device,
            early_stopping=early_stopping,
        )

        start_epoch = checkpoint["epoch"] + 1

        extra_state = checkpoint.get("extra_state", {})

        if "rng_state" in extra_state:
            restore_rng_state(extra_state["rng_state"])

        if "train_generator_state" in extra_state:
            train_generator.set_state(
                extra_state["train_generator_state"].cpu().to(
                    dtype=torch.uint8
                )
            )

        print(
            f"\n[RESUME] Continuing from epoch {start_epoch}"
        )

        if early_stopping.should_stop:
            print(
                "[INFO] Early stopping was already triggered. "
                "No further training will run."
            )
            return

    else:
        save_json(config_path, config)

    if start_epoch > args.epochs:
        print(
            "[INFO] Requested epoch count already completed."
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            epoch=epoch,
        )

        validation_metrics = run_epoch(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        validation_loss = validation_metrics["loss"]

        improved = early_stopping.step(validation_loss)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_loss,
            "validation_accuracy": validation_metrics["accuracy"],
        }

        append_history(history_path, row)

        checkpoints.save(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            validation_loss=validation_loss,
            best_validation_loss=early_stopping.best_loss,
            is_best=improved,
            early_stopping=early_stopping,
            extra_state={
                "rng_state": capture_rng_state(),
                "train_generator_state": train_generator.get_state(),
            },
        )

        print(
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.2%}"
        )
        print(
            f"Val Loss:   {validation_loss:.4f} | "
            f"Val Acc:   {validation_metrics['accuracy']:.2%}"
        )

        if improved:
            print("[BEST] Validation loss improved.")

        if early_stopping.should_stop:
            print(
                "\n[EARLY STOPPING] "
                "Validation loss stopped improving."
            )
            break

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)
    print(f"History: {history_path}")
    print(f"Best model: {checkpoints.best_path}")
    print(f"Last model: {checkpoints.last_path}")


if __name__ == "__main__":
    main()