"""Dataset helpers for folder-based image classification."""

from pathlib import Path

from torchvision import datasets, transforms


def create_image_transform(image_size: int = 224) -> transforms.Compose:
    """Create preprocessing transforms for RGB image classification."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def load_image_folder(
    root_dir: str | Path,
    image_size: int = 224,
) -> datasets.ImageFolder:
    """Load images from subfolders where each subfolder is one class."""
    return datasets.ImageFolder(
        root=Path(root_dir),
        transform=create_image_transform(image_size),
    )
