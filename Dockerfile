# CardioLens API + web UI (D112). Build: docker build -t cardiolens .
# Run:   docker run -p 8000:8000 cardiolens            -> http://localhost:8000
# Opt-in settings (D104, D109–D111) are plain environment variables, e.g.
#   -e CARDIOLENS_API_KEYS=... -e CARDIOLENS_DB_PATH=/data/cardiolens.db -e CARDIOLENS_DB_KEY=...
#   -e CARDIOLENS_AUDIT_LOG=/data/audit.jsonl -v cardiolens-data:/data
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CARDIOLENS_CONFIG_DIR=/app/configs \
    CARDIOLENS_FEEDBACK_PATH=/data/feedback.jsonl \
    CARDIOLENS_CALIBRATOR_PATH=/data/calibrator.json \
    MPLBACKEND=Agg

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --upgrade pip && pip install ".[dicom]" && useradd --create-home --uid 10001 cardiolens \
    && mkdir -p /data && chown cardiolens /data

USER cardiolens
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "cardiolens.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
