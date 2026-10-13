from __future__ import annotations

import os
from pathlib import Path

import torch


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be at least 1.")

        if min_delta < 0:
            raise ValueError("min_delta cannot be negative.")

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, validation_loss: float) -> bool:
        """Update stopping state and return whether loss improved."""

        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            return True

        self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return False

    def state_dict(self) -> dict:
        return {
            "best_loss": self.best_loss,
            "counter": self.counter,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best_loss = state["best_loss"]
        self.counter = state["counter"]
        self.should_stop = state["should_stop"]


class CheckpointManager:
    """Manage best and latest training checkpoints."""

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_path = self.checkpoint_dir / "best_checkpoint.pt"
        self.last_path = self.checkpoint_dir / "last_checkpoint.pt"

    @staticmethod
    def _atomic_save(checkpoint: dict, path: Path) -> None:
        """Save to a temporary file before replacing the target."""

        temporary_path = path.with_suffix(".tmp")

        try:
            torch.save(checkpoint, temporary_path)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def save(
        self,
        *,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        validation_loss: float,
        best_validation_loss: float,
        is_best: bool,
        early_stopping: EarlyStopping | None = None,
        extra_state: dict | None = None,
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": validation_loss,
            "best_validation_loss": best_validation_loss,
        }

        if early_stopping is not None:
            checkpoint["early_stopping_state"] = (
                early_stopping.state_dict()
            )

        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        self._atomic_save(
            checkpoint,
            self.last_path,
        )

        if is_best:
            self._atomic_save(
                checkpoint,
                self.best_path,
            )

    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        best: bool = False,
        device: str | torch.device = "cpu",
        early_stopping: EarlyStopping | None = None,
    ) -> dict:
        checkpoint_path = (
            self.best_path if best else self.last_path
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if optimizer is not None:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if (
            early_stopping is not None
            and "early_stopping_state" in checkpoint
        ):
            early_stopping.load_state_dict(
                checkpoint["early_stopping_state"]
            )

        return checkpoint