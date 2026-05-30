"""Shared utilities for classifier command-line scripts."""

from pathlib import Path
from typing import Any

from typing_extensions import TypedDict

import torch

from image_classifier.model import Classifier


class Checkpoint(TypedDict):
    """Serialized model checkpoint structure."""

    model_state_dict: dict[str, Any]
    class_to_idx: dict[str, int]
    config: dict[str, Any]


def get_device(requested_device: str) -> torch.device:
    """Select the best available device for training or inference."""
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


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


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Checkpoint:
    """Load a checkpoint onto the requested device."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint at {checkpoint_path} is not a dictionary.")
    return checkpoint  # type: ignore[return-value]


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[Classifier, Checkpoint]:
    """Load a classifier and its checkpoint metadata."""
    checkpoint = load_checkpoint(checkpoint_path=checkpoint_path, device=device)
    config = checkpoint["config"]
    model = Classifier(
        in_channels=int(config.get("in_channels", 3)),
        num_classes=int(config["num_classes"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
