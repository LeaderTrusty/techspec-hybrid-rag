from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "TechSpec Hybrid RAG API"
    chunks_path: Path = Path(os.getenv("CHUNKS_PATH", "storage/chunks.jsonl"))
    pdf_dir: Path = Path(os.getenv("PDF_DIR", "data/datasheets"))
    qdrant_path: Path = Path(os.getenv("QDRANT_PATH", "storage/qdrant_db"))
    collection_name: str = os.getenv("QDRANT_COLLECTION", "techspec_datasheets")
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    min_rerank_score: float = float(os.getenv("MIN_RERANK_SCORE", "0.0"))
    skip_startup_init: bool = os.getenv("SKIP_STARTUP_INIT", "0") == "1"


def get_settings() -> Settings:
    return Settings()
