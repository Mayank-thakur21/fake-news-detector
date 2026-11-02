# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Multi-target image:
#   docker build --target backend  -t fake-news-api .
#   docker build --target frontend -t fake-news-ui  .
# ---------------------------------------------------------------------------
ARG PYTHON_VERSION=3.13

FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN groupadd --system app && useradd --system --gid app --home /app app

# ---------------------------------------------------------------------------
FROM base AS backend
COPY requirements/backend.txt requirements/backend.txt
RUN pip install -r requirements/backend.txt
COPY backend/ backend/
COPY ml/ ml/
COPY data/sample/ data/sample/
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p models storage \
    && chown -R app:app /app
USER app
ENV MODEL_DIR=/app/models \
    DB_PATH=/app/storage/predictions.db \
    AUTO_TRAIN=true
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]

# ---------------------------------------------------------------------------
FROM base AS frontend
COPY requirements/frontend.txt requirements/frontend.txt
RUN pip install -r requirements/frontend.txt
COPY frontend/ frontend/
COPY .streamlit/ .streamlit/
RUN chown -R app:app /app
USER app
ENV API_URL=http://backend:8000
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"
CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
