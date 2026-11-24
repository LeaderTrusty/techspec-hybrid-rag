#!/usr/bin/env bash
set -e

# 1. Start FastAPI backend internally on port 8000
uvicorn app.main:app --host 127.0.0.1 --port 8000 &

# 2. Wait up to 60s for FastAPI & Qdrant index to come online
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8000/health > /dev/null; then
    echo "FastAPI backend is healthy."
    break
  fi
  sleep 1
done

# 3. Serve Streamlit UI publicly on port 7860 (Hugging Face Spaces default)
export API_URL="http://127.0.0.1:8000"
exec streamlit run ui/streamlit_app.py \
  --server.port=7860 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
