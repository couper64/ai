"""DataLoader helpers for image classification datasets."""

from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from classifier.dataset import load_image_folder


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a PyTorch DataLoader for an image classification dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def load_image_folder_dataloader(
    root_dir: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Load an image folder dataset and wrap it in a DataLoader."""
    dataset = load_image_folder(root_dir=root_dir, image_size=image_size)
    return create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
