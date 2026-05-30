"""Run inference with a packaged image classifier artifact."""

import argparse
from pathlib import Path

from image_classifier.inference import predict_image


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Predict an image class from a packaged artifact.")
    parser.add_argument("image_path", type=Path, help="Path to an image file.")
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Packaged artifact directory.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of predictions to print.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device to run inference on.",
    )
    return parser.parse_args()


def main() -> None:
    """Predict class probabilities for a single image."""
    args = parse_args()
    predictions = predict_image(
        image_path=args.image_path,
        artifact_dir=args.artifact_dir,
        top_k=args.top_k,
        requested_device=args.device,
    )

    print(f"Predictions for {args.image_path}:")
    for prediction in predictions:
        print(f"  {prediction['class_name']}: {prediction['confidence']:.4f}")


if __name__ == "__main__":
    main()
