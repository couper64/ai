"""Gradio UI that uploads images to the image classifier FastAPI service."""

import os
from io import BytesIO
from typing import Any

import gradio as gr
import httpx
from PIL import Image


API_URL = os.environ.get("IMAGE_CLASSIFIER_API_URL", "http://localhost:8000/predict")
HEALTH_URL = os.environ.get("IMAGE_CLASSIFIER_HEALTH_URL", "http://localhost:8000/health")
DEFAULT_TOP_K = int(os.environ.get("TOP_K", "3"))
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "30"))


def check_api_health() -> str:
    """Return a short status string for the prediction API."""
    try:
        response = httpx.get(HEALTH_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        return f"API unavailable: {exc}"

    status = payload.get("status", "unknown")
    device = payload.get("device", "unknown")
    artifact_dir = payload.get("artifact_dir", "unknown")
    return f"API status: {status} | device: {device} | artifact: {artifact_dir}"


def classify_image(image: Image.Image | None, top_k: int) -> tuple[str, list[list[Any]]]:
    """Send an uploaded image to the classifier API and return formatted results."""
    if image is None:
        return "Upload an image to classify.", []

    if top_k <= 0:
        return "top_k must be greater than 0.", []

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.post(
                API_URL,
                params={"top_k": top_k},
                files={"file": ("upload.jpg", image_bytes, "image/jpeg")},
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip() or str(exc)
        return f"Prediction failed ({exc.response.status_code}): {detail}", []
    except httpx.HTTPError as exc:
        return f"Could not reach the API: {exc}", []

    predictions = payload.get("predictions", [])
    if not predictions:
        return "The API returned no predictions.", []

    rows = [
        [prediction["class_name"], f"{float(prediction['confidence']):.4f}"]
        for prediction in predictions
    ]
    summary = f"Top {len(predictions)} predictions:"
    return summary, rows


def build_ui() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(title="Image Classifier") as demo:
        gr.Markdown(
            "# Image Classifier\n"
            "Upload a photo and classify it using the FastAPI model service."
        )
        gr.Markdown(check_api_health())

        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload a photo")
            results_table = gr.Dataframe(
                headers=["Class", "Confidence"],
                label="Predictions",
                interactive=False,
            )

        top_k_input = gr.Slider(
            minimum=1,
            maximum=10,
            step=1,
            value=DEFAULT_TOP_K,
            label="Top K",
        )
        summary_output = gr.Markdown()
        classify_button = gr.Button("Classify", variant="primary")

        classify_button.click(
            fn=classify_image,
            inputs=[image_input, top_k_input],
            outputs=[summary_output, results_table],
        )
        image_input.change(
            fn=classify_image,
            inputs=[image_input, top_k_input],
            outputs=[summary_output, results_table],
        )

    return demo


def main() -> None:
    """Launch the Gradio frontend."""
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    root_path = os.environ.get("GRADIO_ROOT_PATH")
    demo = build_ui()
    demo.launch(server_name=host, server_port=port, root_path=root_path)


if __name__ == "__main__":
    main()
