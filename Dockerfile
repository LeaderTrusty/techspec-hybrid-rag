FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1

WORKDIR /app

# Install curl for Docker Compose healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for optimal layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run as non-root user (matches Linux container & HF Spaces security best practices)
RUN useradd -m -u 1000 appuser

COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser ui/ ./ui/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser data/ ./data/
COPY --chown=appuser:appuser storage/ ./storage/

USER appuser

EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
