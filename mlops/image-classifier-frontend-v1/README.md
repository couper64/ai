# Image Classifier Streamlit Frontend

Streamlit UI for uploading a photo and classifying it through the FastAPI image classifier service.

This is the v1 frontend. The Gradio v2 frontend lives in `../image-classifier-frontend-v2`.

The frontend does not load the model directly. It sends HTTP requests to the MLOps API.

## Installation

Install directly from GitHub:

```bash
pip install "git+https://github.com/couper64/ai.git#subdirectory=mlops/image-classifier-frontend-v1"
```

This installs the Streamlit UI and HTTP/image dependencies declared in `pyproject.toml`.

## Development Setup

Create and activate the Conda environment. Conda is only used to manage the Python version:

```bash
conda env create -f environment.yml
conda activate image-classifier-frontend-v1
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

Then start the frontend:

```bash
export IMAGE_CLASSIFIER_API_URL=http://localhost:8000/predict
export IMAGE_CLASSIFIER_HEALTH_URL=http://localhost:8000/health
export TOP_K=3

python -m streamlit run src/image_classifier_frontend_v1/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

Open the URL shown by Streamlit, usually `http://localhost:8501`.

## Configuration

```bash
export IMAGE_CLASSIFIER_API_URL=http://localhost:8000/predict
export IMAGE_CLASSIFIER_HEALTH_URL=http://localhost:8000/health
export TOP_K=3
```

## Homeserver / Reverse Proxy

Typical layout:

```text
Browser -> reverse proxy -> Streamlit frontend
Streamlit frontend -> internal FastAPI service
```

Example environment for a homeserver:

```bash
export IMAGE_CLASSIFIER_API_URL=http://image-classifier-api:8000/predict
export IMAGE_CLASSIFIER_HEALTH_URL=http://image-classifier-api:8000/health
python -m streamlit run src/image_classifier_frontend_v1/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

Expose only the frontend publicly if the API should stay internal to your network.

### Subpath Hosting

To host Streamlit behind a subpath such as `https://bananasbravadas.duckdns.org/image-classifer/v1`, set `STREAMLIT_SERVER_BASE_URL_PATH` (default in `docker-compose.yaml` is `image-classifer/v1`):

```bash
STREAMLIT_SERVER_BASE_URL_PATH=image-classifer/v1 docker compose up -d --build
```

For local runs without Docker:

```bash
export STREAMLIT_SERVER_BASE_URL_PATH=image-classifer/v1
python -m streamlit run src/image_classifier_frontend_v1/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

The reverse proxy path and Streamlit base path must match. Websocket proxying is required.

In Nginx Proxy Manager, add a custom location `/image-classifer/v1` to the Streamlit host on port `8501` with websockets enabled.

Example nginx shape:

```nginx
location /image-classifer/v1/ {
    proxy_pass http://127.0.0.1:8501/image-classifer/v1/;
    proxy_http_version 1.1;

    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```
