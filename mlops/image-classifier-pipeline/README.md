# Image Classifier Pipeline

FastAPI service for serving a packaged image classifier artifact.

## Installation

Install the ML package first, then the serving package:

```bash
pip install "git+https://github.com/couper64/ai.git#subdirectory=ml/image-classifier"
pip install "git+https://github.com/couper64/ai.git#subdirectory=mlops/image-classifier-pipeline"
```

The pipeline imports `image_classifier` at runtime, so both packages must be installed in the same environment.

## Development Setup

Create and activate the Conda environment. Conda is only used to manage the Python version:

```bash
conda env create -f environment.yml
conda activate image-classifier-pipeline
```

Install the local ML package and the pipeline package for development:

```bash
./script/install.sh
```

## Test

Install the test extra, then run API smoke tests from the project root (no running server required):

```bash
pip install -e ".[test]"
python -m pytest test
```

The pipeline tests mock model loading; the ML package must be installed in the same environment.

## Run

Point the service at a packaged artifact directory and start Uvicorn:

```bash
export ARTIFACT_DIR=../../ml/image-classifier/output/artifacts/20260530T165256Z
export DEVICE=auto
export TOP_K=3

uvicorn image_classifier_pipeline.api:app --host 0.0.0.0 --port 8000
```

Logs are written to stdout. By default, the service logs startup/shutdown, invalid prediction inputs, and HTTP requests that are slow or return `4xx`/`5xx`.

```bash
export SLOW_REQUEST_MS=50
export LOG_LEVEL=DEBUG
export LOG_COLOR=auto
```

`LOG_LEVEL=DEBUG` also shows successful prediction summaries, including filename, `top_k`, top class, and confidence.
`LOG_COLOR` defaults to `auto`; set it to `true` to force ANSI colors or `false` to disable them.

## Health Check

```bash
curl http://localhost:8000/health
```

## Predict

Upload an image and return top-k predictions:

```bash
curl -X POST "http://localhost:8000/predict?top_k=3" \
  -F "file=@../../ml/image-classifier/data/animals10/prepared/test/cane/OIP-1sZ8l7JY81pPmZZZa3r09AHaJ4.jpeg"
```

The API loads the packaged artifact once at startup, then reuses it for prediction requests.

## Benchmark

Start the API first, then benchmark the `/predict` endpoint with a sample image:

```bash
image-classifier-pipeline-benchmark \
  ../../ml/image-classifier/data/animals10/prepared/test/cane/OIP-1sZ8l7JY81pPmZZZa3r09AHaJ4.jpeg \
  --url http://localhost:8000/predict \
  --requests 50 \
  --warmup 5 \
  --top-k 3 \
  --metrics-output output/benchmark.json
```

If the console script is not installed yet, reinstall the editable package:

```bash
pip install -e . --no-deps
```

You can also run the benchmark module directly:

```bash
python -m image_classifier_pipeline.cli.benchmark \
  ../../ml/image-classifier/data/animals10/prepared/test/cane/OIP-1sZ8l7JY81pPmZZZa3r09AHaJ4.jpeg
```
