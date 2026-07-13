# Python 3.13 to match the pinned deps (numpy 2.5 / scipy 1.18 need >= 3.12).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

# Runtime config. PORT defaults to 7860 so the image runs on Hugging Face
# Spaces with zero extra config; docker-compose overrides it to 5000 locally.
# HOME=/tmp plus world-writable data dirs let the app create its SQLite DB,
# uploads and reports whether the platform runs the container as root or as a
# non-root user (Hugging Face Spaces runs containers as UID 1000).
ENV FLASK_DEBUG=0 \
    DATABASE_PATH=/app/data/database.db \
    MEDIA_DIR=/app/media \
    PRELOAD_MODELS=1 \
    PORT=7860 \
    HOME=/tmp
RUN mkdir -p /app/data /app/media && chmod -R 777 /app/data /app/media

EXPOSE 7860
# 1 worker (models load once), threads for concurrent requests.
CMD ["sh", "-c", "gunicorn -w 1 --threads 4 -t 180 -b 0.0.0.0:${PORT} app:app"]
