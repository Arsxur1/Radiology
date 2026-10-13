# CardioLens web service (FastAPI + UI). Layers 1-5 run inside this container;
# the Layer-6 narrative is fetched from an OpenAI-compatible server named by
# CARDIOLENS_LLM_BASE_URL (docker-compose starts Ollama next to it) or falls
# back to the deterministic stub. No dataset, model weight or key is baked in.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY data/samples ./data/samples
RUN pip install --upgrade pip && pip install -e ".[digitize]"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c \
  "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"
CMD ["uvicorn", "cardiolens.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
