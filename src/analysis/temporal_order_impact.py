from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from src.config.paths import REPORTS_DIR
from src.training.train import build_dataset, build_model


def reorder_sequence(
    features: torch.Tensor,
    mode: str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Reorder temporal frames without changing their features."""

    if mode == "original":
        return features

    if mode == "reversed":
        return torch.flip(features, dims=[1])

    if mode == "shuffled":
        if generator is None:
            raise ValueError("Shuffled mode requires a generator.")

        batch_size, sequence_length, feature_size = features.shape

        permutations = torch.stack(
            [
                torch.randperm(
                    sequence_length,
                    generator=generator,
                )
                for _ in range(batch_size)
            ]
        ).to(features.device)

        indices = permutations.unsqueeze(-1).expand(
            -1,
            -1,
            feature_size,
        )

        return torch.gather(features, dim=1, index=indices)

    raise ValueError(f"Unsupported temporal mode: {mode}")


@torch.no_grad()
def evaluate_temporal_order(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mode: str,
    seed: int,
) -> dict:
    """Evaluate a model and collect per-video predictions."""

    model.eval()

    generator = torch.Generator().manual_seed(seed)

    all_predictions = []
    all_labels = []

    for features, labels in dataloader:
        features = reorder_sequence(
            features,
            mode=mode,
            generator=generator,
        )

        features = features.to(device)

        logits = model(features)
        predictions = logits.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().tolist()
        )
        all_labels.extend(
            labels.tolist()
        )

    predictions_array = np.asarray(all_predictions)
    labels_array = np.asarray(all_labels)

    correct = int(
        np.sum(predictions_array == labels_array)
    )

    return {
        "mode": mode,
        "seed": seed,
        "videos": len(labels_array),
        "correct": correct,
        "accuracy": correct / len(labels_array),
        "predictions": predictions_array,
        "labels": labels_array,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze temporal order sensitivity of PoseLSTM."
    )

    parser.add_argument(
        "--experiment",
        default="pose_coordinates_only_ablation",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="validation",
    )
    parser.add_argument(
        "--checkpoint",
        choices=["best", "last"],
        default="best",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1.")

    device = torch.device(args.device)

    dataset = build_dataset(
        "pose_coordinates_lstm",
        args.split,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
    )

    model = build_model("pose_coordinates_lstm")

    checkpoint_path = (
        REPORTS_DIR.parent
        / "models"
        / args.experiment
        / f"{args.checkpoint}_checkpoint.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    results = []

    for mode in ["original", "reversed"]:
        result = evaluate_temporal_order(
            model=model,
            dataloader=dataloader,
            device=device,
            mode=mode,
            seed=42,
        )
        results.append(result)

    for repeat in range(args.repeats):
        result = evaluate_temporal_order(
            model=model,
            dataloader=dataloader,
            device=device,
            mode="shuffled",
            seed=42 + repeat,
        )
        results.append(result)

    original_predictions = results[0]["predictions"]

    for result in results:
        result["prediction_agreement"] = float(
            np.mean(
                result["predictions"] == original_predictions
            )
        )

    # Arrays are saved separately from the summary CSV.
    prediction_rows = []

    for result in results:
        for video_index, (label, prediction) in enumerate(
            zip(result["labels"], result["predictions"])
        ):
            prediction_rows.append(
                {
                    "mode": result["mode"],
                    "seed": result["seed"],
                    "video_index": video_index,
                    "true_label": int(label),
                    "predicted_label": int(prediction),
                    "agrees_with_original": bool(
                        prediction == original_predictions[video_index]
                    ),
                }
            )

    predictions_df = pd.DataFrame(prediction_rows)

    for result in results:
        del result["predictions"]
        del result["labels"]

    results_df = pd.DataFrame(results)

    output_dir = (
        REPORTS_DIR
        / args.experiment
        / args.split
        / args.checkpoint
        / "temporal_order_impact"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_dir / "results.csv",
        index=False,
    )
    
    predictions_df.to_csv(
        output_dir / "predictions.csv",
        index=False,
    )

    print("\nTEMPORAL ORDER IMPACT ANALYSIS")
    print("=" * 55)

    print(
        results_df[
            ["mode", "seed", "videos", "correct", "accuracy"]
        ].to_string(index=False)
    )

    shuffled = results_df[
        results_df["mode"] == "shuffled"
    ]

    print("\nSUMMARY")
    print("=" * 55)

    print(
        f"Original accuracy: "
        f"{results_df.iloc[0]['accuracy']:.2%}"
    )

    print(
        f"Reversed accuracy: "
        f"{results_df.iloc[1]['accuracy']:.2%}"
    )

    print(
        f"Shuffled mean accuracy: "
        f"{shuffled['accuracy'].mean():.2%}"
    )

    print(
        f"Shuffled std: "
        f"{shuffled['accuracy'].std(ddof=0):.4f}"
    )

    print(
        f"Reversed prediction agreement: "
        f"{results_df.iloc[1]['prediction_agreement']:.2%}"
    )

    print(
        f"Shuffled mean prediction agreement: "
        f"{shuffled['prediction_agreement'].mean():.2%}"
    )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()