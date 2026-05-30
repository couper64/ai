"""Classifier prediction routes."""

import logging
import os
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from image_classifier.inference import ImageClassifierPredictor, Prediction


logger = logging.getLogger(__name__)


ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "output/artifacts/latest"))
DEVICE = os.environ.get("DEVICE", "auto")
DEFAULT_TOP_K = int(os.environ.get("TOP_K", "3"))
COMPILE_MODEL = os.environ.get("COMPILE_MODEL", "false").lower() in {"1", "true", "yes", "on"}
COMPILE_MODE = os.environ.get("COMPILE_MODE", "default")

router = APIRouter(tags=["classifier"])
predictor: ImageClassifierPredictor | None = None


def get_predictor() -> ImageClassifierPredictor:
    """Return the loaded predictor or raise a service error."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")
    return predictor


def load_predictor() -> None:
    """Load the packaged model artifact."""
    global predictor  # noqa: PLW0603
    predictor = ImageClassifierPredictor(
        artifact_dir=ARTIFACT_DIR,
        requested_device=DEVICE,
        compile_model=COMPILE_MODEL,
        compile_mode=COMPILE_MODE,
    )


def unload_predictor() -> None:
    """Release the loaded predictor."""
    global predictor  # noqa: PLW0603
    predictor = None


@router.get("/health")
def health() -> dict[str, str]:
    """Check whether the API is running and the model is loaded."""
    return {
        "status": "ok" if predictor is not None else "loading",
        "artifact_dir": str(ARTIFACT_DIR),
        "device": str(predictor.device) if predictor is not None else DEVICE,
        "compile_model": str(COMPILE_MODEL).lower(),
        "compile_mode": COMPILE_MODE,
    }


@router.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = DEFAULT_TOP_K) -> dict[str, list[Prediction] | str]:
    """Predict top-k classes for an uploaded image."""
    filename = file.filename or ""
    if top_k <= 0:
        logger.warning("prediction_rejected_invalid_top_k filename=%s top_k=%s", filename, top_k)
        raise HTTPException(status_code=400, detail="top_k must be greater than 0.")

    image_bytes = await file.read()
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            predictions = get_predictor().predict_pil_image(image=image, top_k=top_k)
    except UnidentifiedImageError as exc:
        logger.warning("prediction_rejected_invalid_image filename=%s", filename)
        raise HTTPException(status_code=400, detail="Uploaded file is not a supported image.") from exc

    top_prediction = predictions[0] if predictions else None
    if top_prediction is None:
        logger.debug(
            "prediction_completed filename=%s top_k=%s prediction_count=0",
            filename,
            top_k,
        )
    else:
        logger.debug(
            "prediction_completed filename=%s top_k=%s prediction_count=%s top_class=%s top_confidence=%.4f",
            filename,
            top_k,
            len(predictions),
            top_prediction["class_name"],
            top_prediction["confidence"],
        )

    return {
        "filename": filename,
        "predictions": predictions,
    }
