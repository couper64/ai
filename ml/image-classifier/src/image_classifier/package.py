"""Package a trained image classifier checkpoint for inference."""

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from image_classifier.util import count_parameters, get_device, load_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Package an image classifier checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("output/checkpoint/best.pt"),
        help="Path to the checkpoint selected for packaging.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/artifacts"),
        help="Folder where packaged artifacts will be written.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Artifact run ID. Defaults to a UTC timestamp.")
    parser.add_argument("--model-name", type=str, default="image-classifier", help="Model name written to metadata.")
    parser.add_argument("--model-version", type=str, default="0.1.0", help="Model version written to metadata.")
    parser.add_argument("--metrics", type=Path, default=None, help="Optional metrics JSON file to copy into the package.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device used to validate the checkpoint during packaging.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the run folder if it already exists.")
    return parser.parse_args()


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write formatted JSON."""
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_git_sha() -> str | None:
    """Return the current git commit SHA when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def create_run_id() -> str:
    """Create a filesystem-friendly UTC run ID."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    """Create a packaged artifact directory from a training checkpoint."""
    args = parse_args()
    run_id = args.run_id or create_run_id()
    artifact_dir = args.output_dir / run_id

    if artifact_dir.exists() and args.overwrite:
        shutil.rmtree(artifact_dir)
    if artifact_dir.exists():
        raise FileExistsError(f"{artifact_dir} already exists. Use --overwrite to replace it.")
    artifact_dir.mkdir(parents=True)

    device = get_device(args.device)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path=args.checkpoint, device=device)
    config = checkpoint["config"]
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {str(index): class_name for class_name, index in class_to_idx.items()}
    image_size = int(config.get("image_size", 224))

    shutil.copy2(args.checkpoint, artifact_dir / "model.pt")
    write_json(artifact_dir / "class_to_idx.json", class_to_idx)
    write_json(artifact_dir / "idx_to_class.json", idx_to_class)
    write_json(
        artifact_dir / "preprocessing.json",
        {
            "image_size": image_size,
            "resize": [image_size, image_size],
            "color_mode": "RGB",
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
    )
    write_json(
        artifact_dir / "metadata.json",
        {
            "model_name": args.model_name,
            "model_version": args.model_version,
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "task": "image-classification",
            "framework": "pytorch",
            "framework_version": torch.__version__,
            "git_sha": get_git_sha(),
            "checkpoint": str(args.checkpoint),
            "num_parameters": count_parameters(model),
            "config": config,
        },
    )

    if args.metrics is not None:
        shutil.copy2(args.metrics, artifact_dir / "metrics.json")
    else:
        write_json(
            artifact_dir / "metrics.json",
            {
                "metrics_available": False,
                "message": "No metrics file was provided during packaging.",
            },
        )

    print(f"Packaged artifact at {artifact_dir}")


if __name__ == "__main__":
    main()
