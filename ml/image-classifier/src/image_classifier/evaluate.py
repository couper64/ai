"""Evaluate an image classifier checkpoint."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from image_classifier.dataloader import create_dataloader
from image_classifier.dataset import load_image_folder
from image_classifier.util import count_parameters, get_device, load_model_from_checkpoint

ConfusionMatrix = list[list[int]]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate an image classifier checkpoint.")
    parser.add_argument("data_dir", type=Path, help="Evaluation folder with one subfolder per class.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("output/checkpoint/best.pt"),
        help="Path to a model checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--image-size", type=int, default=None, help="Input image size. Defaults to checkpoint config.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--metrics-output", type=Path, default=None, help="Optional path to write metrics as JSON.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device to evaluate on.",
    )
    return parser.parse_args()


def validate_classes(dataset_classes: list[str], checkpoint_classes: list[str]) -> None:
    """Ensure the evaluation dataset uses the checkpoint class order."""
    if dataset_classes != checkpoint_classes:
        raise ValueError(
            "Evaluation dataset classes do not match checkpoint classes. "
            f"dataset={dataset_classes}, checkpoint={checkpoint_classes}"
        )


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, ConfusionMatrix]:
    """Evaluate a model and return overall metrics plus a confusion matrix."""
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    model.eval()
    with torch.inference_mode():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == targets).sum().item()
            total_samples += batch_size

            for target, prediction in zip(targets.tolist(), predictions.tolist(), strict=True):
                confusion_matrix[target][prediction] += 1

    return total_loss / total_samples, total_correct / total_samples, confusion_matrix


def calculate_classification_metrics(confusion_matrix: ConfusionMatrix) -> list[tuple[int, int, float, float, float]]:
    """Calculate support, correct count, precision, recall, and F1 for each class."""
    metrics = []
    num_classes = len(confusion_matrix)
    for class_idx in range(num_classes):
        true_positive = confusion_matrix[class_idx][class_idx]
        false_positive = sum(confusion_matrix[row_idx][class_idx] for row_idx in range(num_classes)) - true_positive
        false_negative = sum(confusion_matrix[class_idx]) - true_positive
        support = true_positive + false_negative

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics.append((support, true_positive, precision, recall, f1_score))
    return metrics


def calculate_random_guessing_accuracy(num_classes: int) -> float:
    """Calculate uniform random guessing accuracy."""
    return 1 / num_classes if num_classes else 0.0


def calculate_majority_class_accuracy(confusion_matrix: ConfusionMatrix) -> float:
    """Calculate accuracy from always predicting the most common true class."""
    class_supports = [sum(row) for row in confusion_matrix]
    total_samples = sum(class_supports)
    return max(class_supports) / total_samples if total_samples else 0.0


def build_metrics_report(
    class_names: list[str],
    loss: float,
    accuracy: float,
    confusion_matrix: ConfusionMatrix,
) -> dict[str, Any]:
    """Build a JSON-serializable evaluation metrics report."""
    per_class = {}
    for class_name, (support, correct, precision, recall, f1_score) in zip(
        class_names,
        calculate_classification_metrics(confusion_matrix),
        strict=True,
    ):
        per_class[class_name] = {
            "support": support,
            "correct": correct,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    return {
        "loss": loss,
        "accuracy": accuracy,
        "random_guessing_accuracy": calculate_random_guessing_accuracy(len(class_names)),
        "majority_class_accuracy": calculate_majority_class_accuracy(confusion_matrix),
        "confusion_matrix": confusion_matrix,
        "classes": class_names,
        "per_class": per_class,
    }


def write_metrics_report(path: Path, metrics_report: dict[str, Any]) -> None:
    """Write evaluation metrics to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_confusion_matrix(class_names: list[str], confusion_matrix: ConfusionMatrix) -> None:
    """Print a readable confusion matrix."""
    label_width = max(len(class_name) for class_name in class_names)
    count_width = max(5, max(len(str(count)) for row in confusion_matrix for count in row))
    print("Confusion matrix (rows=true, columns=predicted):")
    header = " " * (label_width + 2) + " ".join(class_name[:count_width].rjust(count_width) for class_name in class_names)
    print(header)
    for class_name, row in zip(class_names, confusion_matrix, strict=True):
        row_counts = " ".join(str(count).rjust(count_width) for count in row)
        print(f"{class_name.rjust(label_width)}  {row_counts}")


def main() -> None:
    """Evaluate a trained classifier on a held-out dataset."""
    args = parse_args()
    device = get_device(args.device)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path=args.checkpoint, device=device)

    config = checkpoint["config"]
    checkpoint_classes = list(config["classes"])
    image_size = args.image_size if args.image_size is not None else int(config.get("image_size", 224))
    dataset = load_image_folder(root_dir=args.data_dir, image_size=image_size)
    validate_classes(dataset_classes=dataset.classes, checkpoint_classes=checkpoint_classes)

    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    criterion = nn.CrossEntropyLoss()
    loss, accuracy, confusion_matrix = evaluate(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        num_classes=len(checkpoint_classes),
    )

    metrics_report = build_metrics_report(
        class_names=checkpoint_classes,
        loss=loss,
        accuracy=accuracy,
        confusion_matrix=confusion_matrix,
    )
    if args.metrics_output is not None:
        write_metrics_report(path=args.metrics_output, metrics_report=metrics_report)
        print(f"Wrote metrics to {args.metrics_output}")

    print(f"Evaluating on {device} with {len(dataset)} images and {len(checkpoint_classes)} classes.")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"loss={loss:.4f} accuracy={accuracy:.4f}")
    print(
        "Baselines: "
        f"random_guessing_accuracy={metrics_report['random_guessing_accuracy']:.4f} "
        f"majority_class_accuracy={metrics_report['majority_class_accuracy']:.4f}"
    )
    print_confusion_matrix(class_names=checkpoint_classes, confusion_matrix=confusion_matrix)
    print("Per-class metrics:")
    for class_name, (support, correct, precision, recall, f1_score) in zip(
        checkpoint_classes,
        calculate_classification_metrics(confusion_matrix),
        strict=True,
    ):
        print(
            f"  {class_name}: "
            f"precision={precision:.4f} recall={recall:.4f} f1={f1_score:.4f} "
            f"support={support} correct={correct}"
        )


if __name__ == "__main__":
    main()
