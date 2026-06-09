# Image Classifier Gradio Frontend

Gradio UI for uploading a photo and classifying it through the FastAPI image classifier service.

This frontend is intentionally separate from the Streamlit version in `../image-classifier-frontend-v1`.
Both frontends call the same MLOps API over HTTP and do not load the model directly.

## Installation

Install directly from GitHub:

```bash
pip install "git+https://github.com/couper64/ai.git#subdirectory=mlops/image-classifier-frontend-v2"
```

This installs the Gradio UI and HTTP/image dependencies declared in `pyproject.toml`.

## Development Setup

Create and activate the Conda environment. Conda is only used to manage the Python version:

```bash
conda env create -f environment.yml
conda activate image-classifier-frontend-v2
```

Install the project for local development:

```bash
./script/install.sh
```

## Run

Start the FastAPI service first in another terminal:

```bash
cd ../image-classifier-pipeline
export ARTIFACT_DIR=../../ml/image-classifier/output/artifacts/20260530T165256Z
uvicorn image_classifier_pipeline.api:app --host 0.0.0.0 --port 8000
```

Then start the Gradio frontend:

```bash
export IMAGE_CLASSIFIER_API_URL=http://localhost:8000/predict
export IMAGE_CLASSIFIER_HEALTH_URL=http://localhost:8000/health
export TOP_K=3

image-classifier-frontend-v2
```

Open the URL shown by Gradio, usually `http://localhost:7860`.

## Docker

Build and run the Gradio frontend container:

```bash
docker compose up -d --build
```

By default, the container calls the API on the host machine:

```text
http://host.docker.internal:8000/predict
http://host.docker.internal:8000/health
```

Override API URLs or the published port with environment variables:

```bash
IMAGE_CLASSIFIER_API_URL=http://image-classifier-pipeline:8000/predict \
IMAGE_CLASSIFIER_HEALTH_URL=http://image-classifier-pipeline:8000/health \
GRADIO_PORT=7860 \
docker compose up -d --build
```

For subpath hosting behind a reverse proxy, pass `GRADIO_ROOT_PATH`:

```bash
GRADIO_ROOT_PATH=/frontend/v2 docker compose up -d --build
```

Open the app at `http://localhost:7860` unless a reverse proxy maps it elsewhere.

## Configuration

```bash
export IMAGE_CLASSIFIER_API_URL=http://localhost:8000/predict
export IMAGE_CLASSIFIER_HEALTH_URL=http://localhost:8000/health
export TOP_K=3
export HOST=0.0.0.0
export PORT=7860
export GRADIO_ROOT_PATH=
```

## Homeserver / Reverse Proxy

Typical layout:

```text
Browser -> reverse proxy -> Gradio frontend
Gradio frontend -> internal FastAPI service
```

Example environment for a homeserver:

```bash
export IMAGE_CLASSIFIER_API_URL=http://image-classifier-api:8000/predict
export IMAGE_CLASSIFIER_HEALTH_URL=http://image-classifier-api:8000/health
export HOST=0.0.0.0
export PORT=7860
```

Expose only the frontend publicly if the API should stay internal to your network.

### Subpath Hosting

To host Gradio behind a subpath such as `http://localhost/gradio`, set `GRADIO_ROOT_PATH` before starting the app:

```bash
export IMAGE_CLASSIFIER_API_URL=http://image-classifier-api:8000/predict
export IMAGE_CLASSIFIER_HEALTH_URL=http://image-classifier-api:8000/health
export HOST=0.0.0.0
export PORT=7860
export GRADIO_ROOT_PATH=/gradio

image-classifier-frontend-v2
```

The reverse proxy path and Gradio root path must match. Websocket proxying is required.

Example nginx shape:

```nginx
location /gradio/ {
    proxy_pass http://127.0.0.1:7860/gradio/;
    proxy_http_version 1.1;

    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```
