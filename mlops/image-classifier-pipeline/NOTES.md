# Notes

## Summary

This project serves a packaged PyTorch animal image classifier through a FastAPI API.
It is the MLOps serving layer for the `image-classifier` ML package.

The service loads a packaged artifact at startup using `ARTIFACT_DIR`, keeps the model in memory, and exposes:

- `GET /health` for service/model status.
- `POST /predict` for image upload and top-k class predictions.
- Swagger/OpenAPI docs from FastAPI.

The current artifact is:

- Run ID: `20260530T165256Z`
- Model: small custom CNN
- Framework: PyTorch `2.12.0+cu130`
- Parameters: `94,986`
- Classes: `10` Animals-10 classes

## Architecture

The project is split by responsibility:

- `project/ai/ml/image-classifier`: ML package. Owns model definition, training, evaluation, packaging, and reusable inference.
- `project/ai/mlops/image-classifier-pipeline`: MLOps package. Owns FastAPI serving, API tests, benchmarks, and operational logging.

Runtime flow:

```text
client image upload
-> FastAPI /predict
-> ImageClassifierPredictor
-> packaged model artifact
-> JSON predictions
```

The packaged artifact includes:

- `model.pt`
- `metadata.json`
- `preprocessing.json`
- `class_to_idx.json`
- `idx_to_class.json`
- `metrics.json`

## Decisions

- Kept ML and MLOps packages separate to mirror the handoff between ML engineer and MLOps engineer.
- Used FastAPI because it provides simple HTTP serving, Swagger docs, file upload support, and test tooling.
- Loaded the model once during FastAPI lifespan startup instead of loading on each request.
- Used routers (`router/classifier.py`) to keep classifier routes separate from app setup (`api.py`).
- Kept `torch.compile` optional through `COMPILE_MODEL` and `COMPILE_MODE`; benchmark results did not justify enabling it by default for this tiny CNN.
- Added API smoke tests and ML inference smoke tests, then wired them into GitHub Actions CI on `dev` and PRs.
- Added operational logs but kept them intentionally quiet: errors, slow requests, invalid inputs, and optional debug prediction summaries.

## Metrics and Evidence

Model quality from the packaged artifact:

- Test accuracy: `0.4994`
- Loss: `1.4606`
- Random guessing baseline: `0.1000`
- Majority-class baseline: `0.1855`
- Interpretation: the model beats naive baselines but remains weak; the custom CNN likely underfits and is below what transfer learning could achieve.

Service benchmark on local CUDA, sequential requests, one sample image:

| Mode | Requests | Warmup | Mean | p95 | Throughput | Failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Eager PyTorch | 1000 | 100 | `1.87 ms` | `2.15 ms` | `533.79 req/s` | `0` |
| `torch.compile(reduce-overhead)` | 1000 | 100 | `1.88 ms` | `2.19 ms` | `531.87 req/s` | `0` |

Decision from optimization experiment:

- `torch.compile` did not provide meaningful steady-state improvement for this workload.
- Keep `COMPILE_MODEL=false` by default.

Testing evidence:

- ML package tests: `2 passed`.
- MLOps package tests: `4 passed`.
- GitHub Actions workflow: `Test AI Projects` passes on `dev`.

Operational evidence:

- `/health` returns artifact, device, and compile settings.
- Slow/error request middleware logs request status and duration.
- Prediction route logs invalid inputs at warning level.
- Successful prediction summaries are available with `LOG_LEVEL=DEBUG`.

## Known Limitations

- Accuracy is modest; the model should be treated as a learning artifact, not a production-grade classifier.
- Benchmarks were local, sequential, and based on a single repeated image. They do not represent internet latency, concurrent users, or production infrastructure.
- There is no authentication, rate limiting, upload-size guard, or production deployment hardening.
- Operational logging is minimal and stdout-based; there is no Prometheus/Grafana/Datadog integration yet.
- Data drift is not implemented. Current monitoring can observe service behavior, but not real input-distribution drift against training data.
- `torch.compile` can make first-request latency slower due to compilation/warmup.

## Responsible ML Notes

- This is a low-risk demo domain: animal image classification.
- The model can misclassify images and should not be used for safety-critical decisions.
- Do not log raw image bytes or store uploaded images by default.
- Logging top class and confidence is useful for monitoring, but it should be reconsidered if inputs become sensitive or user-identifiable.
- Low confidence or unusual prediction distributions should be treated as signals to inspect data quality or retrain.

## Runbook

Install for development:

```bash
conda env create -f environment.yml
conda activate image-classifier-pipeline
./script/install.sh
```

Run the API:

```bash
export ARTIFACT_DIR=../../ml/image-classifier/output/artifacts/20260530T165256Z
export DEVICE=auto
export TOP_K=3
export COMPILE_MODEL=false
export SLOW_REQUEST_MS=50
export LOG_LEVEL=INFO
export LOG_COLOR=auto

uvicorn image_classifier_pipeline.api:app --host 0.0.0.0 --port 8000
```

Check health:

```bash
curl http://localhost:8000/health
```

Run one prediction:

```bash
curl -X POST "http://localhost:8000/predict?top_k=3" \
  -F "file=@../../ml/image-classifier/data/animals10/prepared/test/cane/OIP-1sZ8l7JY81pPmZZZa3r09AHaJ4.jpeg"
```

Run tests:

```bash
pip install -e ".[test]"
python -m pytest test
```

Benchmark the service:

```bash
image-classifier-pipeline-benchmark \
  ../../ml/image-classifier/data/animals10/prepared/test/cane/OIP-1sZ8l7JY81pPmZZZa3r09AHaJ4.jpeg \
  --url http://localhost:8000/predict \
  --requests 1000 \
  --warmup 100 \
  --top-k 3 \
  --metrics-output output/benchmark-eager.json
```

Rollback:

```bash
export ARTIFACT_DIR=../../ml/image-classifier/output/artifacts/<previous-run-id>
uvicorn image_classifier_pipeline.api:app --host 0.0.0.0 --port 8000
```

## Next Steps

- Try transfer learning in the ML package to improve accuracy.
- Add a small scheduled evaluation workflow if model quality monitoring becomes important.
- Add upload-size limits and authentication if exposing the API beyond local/dev use.
- Add production metrics/alerts only when stdout logs are no longer enough.