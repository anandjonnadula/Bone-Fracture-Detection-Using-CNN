# 3.13 to match the pinned deps: numpy 2.5 / scipy 1.18 require Python >= 3.12.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

# Data lives on named volumes (see docker-compose.yml); the same code runs
# bare-metal with the in-repo defaults.
ENV FLASK_DEBUG=0 \
    DATABASE_PATH=/app/data/database.db \
    MEDIA_DIR=/app/media \
    PRELOAD_MODELS=1 \
    PORT=5000
RUN mkdir -p /app/data /app/media

EXPOSE 5000
# 1 worker (models load once), threads for concurrent requests. $PORT makes
# the same image deployable to Hugging Face Spaces (which expects 7860).
CMD ["sh", "-c", "gunicorn -w 1 --threads 4 -t 180 -b 0.0.0.0:${PORT} app:app"]
