FROM python:3.11-slim AS build

WORKDIR /build

RUN python -m pip install --upgrade pip build

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m build --wheel --outdir /dist


FROM python:3.11-slim AS deploy

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

RUN python -m venv /opt/venv \
    && python -m pip install --upgrade pip \
    && useradd --create-home --shell /usr/sbin/nologin appuser

COPY --from=build /dist/*.whl /tmp/
RUN python -m pip install /tmp/*.whl \
    && rm -rf /tmp/*.whl /root/.cache/pip

COPY --chown=appuser:appuser src ./src

USER appuser

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "src/image_classifier_frontend_v1/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
