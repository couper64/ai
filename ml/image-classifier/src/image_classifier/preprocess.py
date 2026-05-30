"""Prepare image classification datasets."""

import argparse
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Split an image folder into train/validation/test sets.")
    parser.add_argument("source_dir", type=Path, help="Input folder with one subfolder per class.")
    parser.add_argument("output_dir", type=Path, help="Output folder for train, validation, and test splits.")
    parser.add_argument("--train-split", type=float, default=0.7, help="Fraction of images used for training.")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.15,
        help="Fraction of images used for validation.",
    )
    parser.add_argument("--test-split", type=float, default=0.15, help="Fraction of images used for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output folder before writing the split dataset.",
    )
    args = parser.parse_args()

    split_total = args.train_split + args.validation_split + args.test_split
    if abs(split_total - 1.0) > 1e-6:
        parser.error("--train-split, --validation-split, and --test-split must sum to 1.0.")
    if min(args.train_split, args.validation_split, args.test_split) <= 0.0:
        parser.error("All split fractions must be greater than 0.")
    return args


def list_images(class_dir: Path) -> list[Path]:
    """List image files in a class directory."""
    extensions = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    return sorted(path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions)


def split_class_images(
    images: list[Path],
    train_split: float,
    validation_split: float,
    rng: random.Random,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split class images into train, validation, and test lists."""
    shuffled = images.copy()
    rng.shuffle(shuffled)

    train_count = int(len(shuffled) * train_split)
    validation_count = int(len(shuffled) * validation_split)
    train_images = shuffled[:train_count]
    validation_images = shuffled[train_count : train_count + validation_count]
    test_images = shuffled[train_count + validation_count :]

    if not train_images or not validation_images or not test_images:
        raise ValueError("Each class must have enough images to populate all splits.")
    return train_images, validation_images, test_images


def copy_images(images: list[Path], destination_dir: Path) -> None:
    """Copy images into a split class directory."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        shutil.copy2(image_path, destination_dir / image_path.name)


def main() -> None:
    """Split a folder-based image dataset into fixed subsets."""
    args = parse_args()
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"{args.output_dir} already exists and is not empty. Use --overwrite to replace it.")

    class_dirs = sorted(path for path in args.source_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class folders found in {args.source_dir}.")

    rng = random.Random(args.seed)
    total_counts = {"train": 0, "validation": 0, "test": 0}
    for class_dir in class_dirs:
        images = list_images(class_dir)
        if not images:
            raise ValueError(f"No image files found in {class_dir}.")
        train_images, validation_images, test_images = split_class_images(
            images=images,
            train_split=args.train_split,
            validation_split=args.validation_split,
            rng=rng,
        )

        split_images = {
            "train": train_images,
            "validation": validation_images,
            "test": test_images,
        }
        for split_name, image_paths in split_images.items():
            copy_images(image_paths, args.output_dir / split_name / class_dir.name)
            total_counts[split_name] += len(image_paths)

    print(
        f"Created dataset split at {args.output_dir}: "
        f"train={total_counts['train']}, validation={total_counts['validation']}, test={total_counts['test']}"
    )


if __name__ == "__main__":
    main()
