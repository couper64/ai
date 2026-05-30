"""Smoke tests for packaged artifact inference."""

import json
from pathlib import Path

import torch
from PIL import Image

from image_classifier.inference import ImageClassifierPredictor
from image_classifier.model import Classifier
from image_classifier.util import save_checkpoint


def write_json(path: Path, data: dict[str, object]) -> None:
    """Write a test JSON sidecar."""
    path.write_text(json.dumps(data), encoding="utf-8")


def create_artifact(tmp_path: Path) -> Path:
    """Create a minimal packaged artifact for inference tests."""
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    class_to_idx = {"cat": 0, "dog": 1}
    model = Classifier(in_channels=3, num_classes=len(class_to_idx))
    save_checkpoint(
        output_path=artifact_dir / "model.pt",
        model=model,
        class_to_idx=class_to_idx,
        config={
            "classes": list(class_to_idx),
            "image_size": 32,
            "in_channels": 3,
            "num_classes": len(class_to_idx),
        },
    )
    write_json(artifact_dir / "class_to_idx.json", class_to_idx)
    write_json(artifact_dir / "idx_to_class.json", {"0": "cat", "1": "dog"})
    write_json(
        artifact_dir / "preprocessing.json",
        {
            "image_size": 32,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
    )
    return artifact_dir


def test_predictor_returns_top_k_predictions(tmp_path: Path) -> None:
    """A packaged artifact can be loaded and used for one image prediction."""
    artifact_dir = create_artifact(tmp_path)
    predictor = ImageClassifierPredictor(artifact_dir=artifact_dir, requested_device="cpu")

    image = Image.new("RGB", (48, 48), color=(120, 80, 40))
    predictions = predictor.predict_pil_image(image=image, top_k=2)

    assert len(predictions) == 2
    assert {prediction["class_name"] for prediction in predictions} == {"cat", "dog"}
    assert all(0.0 <= prediction["confidence"] <= 1.0 for prediction in predictions)


def test_predictor_compiles_model_when_requested(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The compile flag delegates model compilation to torch.compile."""
    artifact_dir = create_artifact(tmp_path)
    compiled_models: list[torch.nn.Module] = []

    def fake_compile(model: torch.nn.Module, *, mode: str) -> torch.nn.Module:
        assert mode == "reduce-overhead"
        compiled_models.append(model)
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)

    predictor = ImageClassifierPredictor(
        artifact_dir=artifact_dir,
        requested_device="cpu",
        compile_model=True,
        compile_mode="reduce-overhead",
    )

    assert predictor.compile_model is True
    assert predictor.compile_mode == "reduce-overhead"
    assert compiled_models == [predictor.model]
