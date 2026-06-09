FROM python:3.11-slim AS build

WORKDIR /build

RUN python -m pip install --upgrade pip build

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m build --wheel --outdir /dist


FROM python:3.11-slim AS deploy

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

RUN python -m venv /opt/venv \
    && python -m pip install --upgrade pip \
    && useradd --create-home --shell /usr/sbin/nologin appuser

COPY --from=build /dist/*.whl /tmp/
RUN python -m pip install /tmp/*.whl \
    && rm -rf /tmp/*.whl /root/.cache/pip

USER appuser

EXPOSE 7860

CMD ["image-classifier-frontend-v2"]
