from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.settings import ACTION_CLASSES
from src.config.paths import MODELS_DIR, REPORTS_DIR
from src.training.train import build_model, build_dataset
from src.training.metrics import calculate_classification_metrics


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


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    model.eval()

    all_true = []
    all_pred = []
    all_probabilities = []

    with torch.inference_mode():
        for frames, labels in tqdm(
            dataloader,
            desc="Evaluating",
            unit="batch",
        ):
            frames = frames.to(device)

            logits = model(frames)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            all_true.append(labels.numpy())
            all_pred.append(predictions.cpu().numpy())
            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_probabilities),
    )


def save_results(
    output_dir: Path,
    metrics: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    serializable_metrics = {
        key: value
        for key, value in metrics.items()
        if key != "confusion_matrix"
    }

    with (output_dir / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable_metrics,
            file,
            indent=2,
        )

    np.save(
        output_dir / "confusion_matrix.npy",
        metrics["confusion_matrix"],
    )

    with (output_dir / "predictions.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "true_label",
                "predicted_label",
                "true_class",
                "predicted_class",
                "confidence",
            ]
        )

        for true_label, predicted_label, probs in zip(
            y_true,
            y_pred,
            probabilities,
        ):
            writer.writerow(
                [
                    int(true_label),
                    int(predicted_label),
                    ACTION_CLASSES[int(true_label)],
                    ACTION_CLASSES[int(predicted_label)],
                    float(np.max(probs)),
                ]
            )


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Evaluate a trained human action recognition model."
    )
    
    parser.add_argument(
        "--model",
        choices=["cnn_lstm", "pose_lstm", "pose_mask_lstm", "pose_coordinates_lstm"],
        default="cnn_lstm",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="cnn_lstm_baseline",
    )

    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
    )

    parser.add_argument(
        "--checkpoint",
        choices=["best", "last"],
        default="best",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
    )

    args = parser.parse_args()

    device = resolve_device(args.device)

    checkpoint_path = (
        MODELS_DIR
        / args.experiment
        / f"{args.checkpoint}_checkpoint.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = build_model(args.model)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    
    dataset = build_dataset(
        model_name=args.model,
        split=args.split,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("=" * 60)
    print("HUMAN ACTION RECOGNITION — EVALUATION")
    print("=" * 60)
    print(f"Model:      {args.model}")
    print(f"Experiment: {args.experiment}")
    print(f"Split:      {args.split}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Epoch:      {checkpoint['epoch']}")
    print(f"Device:     {device}")
    print(f"Videos:     {len(dataset)}")
    print("=" * 60)

    y_true, y_pred, probabilities = evaluate_model(
        model,
        dataloader,
        device,
    )

    metrics = calculate_classification_metrics(
        y_true,
        y_pred,
        ACTION_CLASSES,
    )

    output_dir = (
        REPORTS_DIR
        / args.experiment
        / args.split
        / args.checkpoint
    )

    save_results(
        output_dir,
        metrics,
        y_true,
        y_pred,
        probabilities,
    )

    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Precision Macro: {metrics['precision_macro']:.4f}")
    print(f"Recall Macro:    {metrics['recall_macro']:.4f}")
    print(f"F1 Macro:        {metrics['f1_macro']:.4f}")

    print()
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])

    print()
    print("Per-Class Performance:")

    for class_name in ACTION_CLASSES:
        class_metrics = metrics["classification_report"][class_name]

        print(
            f"{class_name:15s} "
            f"Precision: {class_metrics['precision']:.3f} | "
            f"Recall: {class_metrics['recall']:.3f} | "
            f"F1: {class_metrics['f1-score']:.3f}"
        )

    print()
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()