"""Streamlit v1 UI that uploads images to the image classifier FastAPI service."""

import os
from io import BytesIO

import httpx
from PIL import Image
import streamlit as st


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


def classify_image(image: Image.Image, filename: str, top_k: int) -> dict[str, object]:
    """Send an uploaded image to the classifier API and return the response payload."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        response = client.post(
            API_URL,
            params={"top_k": top_k},
            files={"file": (filename, image_bytes, "image/jpeg")},
        )
        response.raise_for_status()
        return response.json()


def render_app() -> None:
    """Render the Streamlit app."""
    st.set_page_config(page_title="Image Classifier", page_icon="image")
    st.title("Image Classifier")
    st.write("Upload a photo and classify it using the FastAPI model service.")

    with st.sidebar:
        st.header("API Settings")
        st.code(API_URL, language="text")
        top_k = st.slider("Top K", min_value=1, max_value=10, value=DEFAULT_TOP_K, step=1)

        if st.button("Check API Health"):
            health_status = check_api_health()
            if health_status.startswith("API unavailable"):
                st.error(health_status)
            else:
                st.success(health_status)

    uploaded_file = st.file_uploader(
        "Choose a photo",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
        help="The image is sent to the FastAPI service for prediction.",
    )

    if uploaded_file is None:
        st.info("Upload a photo to classify.")
        return

    image = Image.open(uploaded_file)
    st.image(image, caption=uploaded_file.name, width="stretch")

    if not st.button("Classify", type="primary"):
        return

    try:
        payload = classify_image(image=image, filename=uploaded_file.name, top_k=top_k)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip() or str(exc)
        st.error(f"Prediction failed ({exc.response.status_code}): {detail}")
        return
    except httpx.HTTPError as exc:
        st.error(f"Could not reach the API: {exc}")
        return
    except ValueError as exc:
        st.error(str(exc))
        return

    predictions = payload.get("predictions", [])
    if not isinstance(predictions, list) or not predictions:
        st.warning("The API returned no predictions.")
        return

    st.subheader(f"Top {len(predictions)} predictions")
    st.dataframe(
        [
            {
                "Class": prediction["class_name"],
                "Confidence": f"{float(prediction['confidence']):.4f}",
            }
            for prediction in predictions
        ],
        width="stretch",
        hide_index=True,
    )

if __name__ == "__main__":
    render_app()
