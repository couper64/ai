"""Command-line training script for image classification."""

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from classifier.dataloader import create_dataloader
from classifier.dataset import load_image_folder
from classifier.model import Classifier


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train an image classifier.")
    parser.add_argument("data_dir", type=Path, help="Folder with one subfolder per class.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/checkpoint"),
        help="Folder where best.pt and last.pt checkpoints will be saved.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of images to reserve for validation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device to train on.",
    )
    args = parser.parse_args()
    if not 0.0 < args.validation_split < 1.0:
        parser.error("--validation-split must be greater than 0 and less than 1.")
    return args


def get_device(requested_device: str) -> torch.device:
    """Select the best available training device."""
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: Classifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch and return average loss and accuracy."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def validate_one_epoch(
    model: Classifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate the model on the validation set."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def save_checkpoint(
    output_path: Path,
    model: Classifier,
    class_to_idx: dict[str, int],
    config: dict[str, Any],
) -> None:
    """Save model weights and class metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "config": config,
        },
        output_path,
    )


def main() -> None:
    """Train the classifier from a folder of labeled images."""
    args = parse_args()
    device = get_device(args.device)

    dataset = load_image_folder(root_dir=args.data_dir, image_size=args.image_size)
    validation_size = int(len(dataset) * args.validation_split)
    training_size = len(dataset) - validation_size
    if training_size == 0 or validation_size == 0:
        raise ValueError("Dataset is too small for the requested validation split.")

    generator = torch.Generator().manual_seed(args.seed)
    training_dataset, validation_dataset = random_split(
        dataset,
        lengths=(training_size, validation_size),
        generator=generator,
    )
    training_dataloader = create_dataloader(
        dataset=training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_dataloader = create_dataloader(
        dataset=validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = Classifier(in_channels=3, num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    checkpoint_config = {
        "in_channels": 3,
        "num_classes": len(dataset.classes),
        "image_size": args.image_size,
        "classes": dataset.classes,
        "validation_split": args.validation_split,
        "seed": args.seed,
    }
    best_accuracy = 0.0

    print(
        f"Training on {device} with {training_size} train images, "
        f"{validation_size} validation images, and {len(dataset.classes)} classes."
    )
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=training_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        validation_loss, validation_accuracy = validate_one_epoch(
            model=model,
            dataloader=validation_dataloader,
            criterion=criterion,
            device=device,
        )
        print(
            f"epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4f} train_accuracy={train_accuracy:.4f} "
            f"validation_loss={validation_loss:.4f} validation_accuracy={validation_accuracy:.4f}"
        )

        save_checkpoint(
            output_path=args.output_dir / "last.pt",
            model=model,
            class_to_idx=dataset.class_to_idx,
            config=checkpoint_config,
        )
        if validation_accuracy >= best_accuracy:
            best_accuracy = validation_accuracy
            save_checkpoint(
                output_path=args.output_dir / "best.pt",
                model=model,
                class_to_idx=dataset.class_to_idx,
                config=checkpoint_config,
            )

    print(f"Saved best checkpoint to {args.output_dir / 'best.pt'}")
    print(f"Saved last checkpoint to {args.output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
