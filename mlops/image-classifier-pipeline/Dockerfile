FROM python:3.11-slim AS build

WORKDIR /build

RUN python -m pip install --upgrade pip build

COPY ml/image-classifier/pyproject.toml ml/image-classifier/README.md ./ml/image-classifier/
COPY ml/image-classifier/src ./ml/image-classifier/src

COPY mlops/image-classifier-pipeline/pyproject.toml mlops/image-classifier-pipeline/README.md ./mlops/image-classifier-pipeline/
COPY mlops/image-classifier-pipeline/src ./mlops/image-classifier-pipeline/src

RUN python -m build --wheel --outdir /dist ./ml/image-classifier \
    && python -m build --wheel --outdir /dist ./mlops/image-classifier-pipeline


FROM python:3.11-slim AS deploy

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARTIFACT_DIR=/artifacts \
    DEVICE=auto \
    TOP_K=3 \
    LOG_LEVEL=INFO \
    LOG_COLOR=auto \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

RUN python -m venv /opt/venv \
    && python -m pip install --upgrade pip \
    && useradd --create-home --shell /usr/sbin/nologin appuser

COPY --from=build /dist/*.whl /tmp/
RUN python -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install /tmp/*.whl \
    && rm -rf /tmp/*.whl /root/.cache/pip

USER appuser

EXPOSE 8000

CMD ["uvicorn", "image_classifier_pipeline.api:app", "--host", "0.0.0.0", "--port", "8000"]
