FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/appuser \
    HF_HOME=/home/appuser/.cache/huggingface \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1

WORKDIR /app

# System dependency for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to keep the image slim and build fast
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces runs containers as UID 1000
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/storage /app/data/datasheets /home/appuser/.cache/huggingface \
    && chown -R 1000:1000 /app /home/appuser

COPY --chown=1000:1000 . /app

USER appuser

# Pre-cache public embedding and cross-encoder weights at build time
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 7860

CMD ["bash", "start.sh"]
