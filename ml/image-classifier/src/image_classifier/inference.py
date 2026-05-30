"""Reusable inference logic for packaged image classifier artifacts."""

import json
from pathlib import Path
from typing import Any

from typing_extensions import TypedDict

import torch
from PIL import Image
from PIL.Image import Image as PILImage
from torchvision import transforms

from image_classifier.util import get_device, load_model_from_checkpoint


class Prediction(TypedDict):
    """Single class prediction."""

    class_name: str
    confidence: float


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def create_prediction_transform(preprocessing: dict[str, Any]) -> transforms.Compose:
    """Create the inference transform from artifact preprocessing metadata."""
    image_size = int(preprocessing["image_size"])
    mean = tuple(float(value) for value in preprocessing["normalize_mean"])
    std = tuple(float(value) for value in preprocessing["normalize_std"])
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_image_tensor(image_path: Path, transform: transforms.Compose) -> torch.Tensor:
    """Load and transform one image into a batched tensor."""
    with Image.open(image_path) as image:
        return transform_image(image=image, transform=transform)


def transform_image(image: PILImage, transform: transforms.Compose) -> torch.Tensor:
    """Transform one image into a batched tensor."""
    rgb_image = image.convert("RGB")
    return transform(rgb_image).unsqueeze(0)


class ImageClassifierPredictor:
    """Reusable predictor for a packaged image classifier artifact."""

    def __init__(
        self,
        artifact_dir: Path,
        requested_device: str = "auto",
        compile_model: bool = False,
        compile_mode: str = "default",
    ) -> None:
        self.artifact_dir = artifact_dir
        self.device = get_device(requested_device)
        self.compile_model = compile_model
        self.compile_mode = compile_mode
        self.model, _checkpoint = load_model_from_checkpoint(
            checkpoint_path=artifact_dir / "model.pt",
            device=self.device,
        )
        if compile_model:
            self.model = torch.compile(self.model, mode=compile_mode)
        self.preprocessing = read_json(artifact_dir / "preprocessing.json")
        self.idx_to_class = read_json(artifact_dir / "idx_to_class.json")
        self.transform = create_prediction_transform(self.preprocessing)

    def predict_tensor(self, image_tensor: torch.Tensor, top_k: int = 3) -> list[Prediction]:
        """Predict top-k classes from a preprocessed image tensor."""
        image_tensor = image_tensor.to(self.device)
        with torch.inference_mode():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
            effective_top_k = min(top_k, probabilities.size(1))
            top_probabilities, top_indices = probabilities.topk(effective_top_k, dim=1)

        predictions: list[Prediction] = []
        for probability, index in zip(top_probabilities[0].tolist(), top_indices[0].tolist(), strict=True):
            predictions.append(
                {
                    "class_name": self.idx_to_class[str(index)],
                    "confidence": float(probability),
                }
            )
        return predictions

    def predict_pil_image(self, image: PILImage, top_k: int = 3) -> list[Prediction]:
        """Predict top-k classes for a PIL image."""
        return self.predict_tensor(image_tensor=transform_image(image=image, transform=self.transform), top_k=top_k)

    def predict_image_path(self, image_path: Path, top_k: int = 3) -> list[Prediction]:
        """Predict top-k classes for an image path."""
        image_tensor = load_image_tensor(image_path=image_path, transform=self.transform)
        return self.predict_tensor(image_tensor=image_tensor, top_k=top_k)


def predict_image(
    image_path: Path,
    artifact_dir: Path,
    top_k: int = 3,
    requested_device: str = "auto",
    compile_model: bool = False,
    compile_mode: str = "default",
) -> list[Prediction]:
    """Predict top-k classes for a single image from a packaged artifact."""
    predictor = ImageClassifierPredictor(
        artifact_dir=artifact_dir,
        requested_device=requested_device,
        compile_model=compile_model,
        compile_mode=compile_mode,
    )
    return predictor.predict_image_path(image_path=image_path, top_k=top_k)
