"""Smoke tests for the image classifier API."""

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from image_classifier_pipeline.api import app
from image_classifier_pipeline.router import classifier


class FakePredictor:
    """Small predictor double for API tests."""

    device = "cpu"

    def __init__(
        self,
        artifact_dir,
        requested_device: str = "auto",
        compile_model: bool = False,
        compile_mode: str = "default",
    ) -> None:
        self.artifact_dir = artifact_dir
        self.requested_device = requested_device
        self.compile_model = compile_model
        self.compile_mode = compile_mode

    def predict_pil_image(self, image: Image.Image, top_k: int = 3) -> list[dict[str, float | str]]:
        return [
            {"class_name": "cat", "confidence": 0.7},
            {"class_name": "dog", "confidence": 0.3},
        ][:top_k]


def create_image_bytes() -> bytes:
    """Create a tiny in-memory PNG upload."""
    buffer = BytesIO()
    Image.new("RGB", (16, 16), color=(20, 40, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def create_client(monkeypatch) -> TestClient:  # type: ignore[no-untyped-def]
    """Create a test client with the model predictor mocked out."""
    classifier.predictor = None
    monkeypatch.setattr(classifier, "ImageClassifierPredictor", FakePredictor)
    return TestClient(app)


def test_health_reports_loaded_predictor(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Health returns service metadata after startup loads the predictor."""
    with create_client(monkeypatch) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["device"] == "cpu"


def test_predict_returns_predictions(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Predict accepts an image upload and returns top-k predictions."""
    with create_client(monkeypatch) as client:
        response = client.post(
            "/predict",
            params={"top_k": 1},
            files={"file": ("sample.png", create_image_bytes(), "image/png")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "sample.png"
    assert payload["predictions"] == [{"class_name": "cat", "confidence": 0.7}]


def test_predict_rejects_invalid_image(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Predict returns a client error for non-image uploads."""
    with create_client(monkeypatch) as client:
        response = client.post(
            "/predict",
            files={"file": ("not-image.txt", b"not an image", "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a supported image."


def test_predict_rejects_invalid_top_k(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Predict validates top_k before running inference."""
    with create_client(monkeypatch) as client:
        response = client.post(
            "/predict",
            params={"top_k": 0},
            files={"file": ("sample.png", create_image_bytes(), "image/png")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "top_k must be greater than 0."
