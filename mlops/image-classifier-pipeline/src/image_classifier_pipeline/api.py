"""FastAPI service for the packaged image classifier."""

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from image_classifier_pipeline.logging_config import configure_logging
from image_classifier_pipeline.router import classifier


FAVICON_PATH = Path(__file__).parent / "favicon.ico"
SLOW_REQUEST_MS = float(os.environ.get("SLOW_REQUEST_MS", "50"))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the model on startup and release it on shutdown."""
    configure_logging()
    classifier.load_predictor()
    logger.info(
        "service_started artifact_dir=%s device=%s compile_model=%s compile_mode=%s",
        classifier.ARTIFACT_DIR,
        classifier.DEVICE,
        classifier.COMPILE_MODEL,
        classifier.COMPILE_MODE,
    )
    yield
    classifier.unload_predictor()
    logger.info("service_stopped")


app = FastAPI(title="Image Classifier API", version="0.1.0", lifespan=lifespan)

app.include_router(classifier.router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log failed and slow HTTP requests."""
    if request.url.path == "/favicon.ico":
        return await call_next(request)

    started_at = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.exception(
            "request_failed method=%s path=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = (time.perf_counter() - started_at) * 1000
    if response.status_code >= 400 or duration_ms > SLOW_REQUEST_MS:
        logger.info(
            "request_completed method=%s path=%s status_code=%s duration_ms=%.2f slow_request_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            SLOW_REQUEST_MS,
        )
    return response


class RootResponse(BaseModel):
    message: str
    date: str
    time: str
    timezone: str


@app.get("/", response_model=RootResponse)
async def root() -> RootResponse:
    """Return a short welcome message and current UTC time."""
    now = datetime.now(timezone.utc)
    return RootResponse(
        message="Welcome to a web service developed by couper64!",
        date=now.strftime("%Y-%m-%d"),
        time=now.strftime("%H:%M:%S"),
        timezone=now.strftime("%Z%z"),
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    """Serve favicon for browsers; omitted from OpenAPI schema."""
    if not FAVICON_PATH.is_file():
        raise HTTPException(status_code=404, detail="favicon not found")
    return FileResponse(FAVICON_PATH)
