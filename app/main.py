from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import Depends, FastAPI, HTTPException, Request

from app.config import get_settings
from app.rag.generator import GroundedRAGPipeline
from app.rag.ingest import ingest_directory, load_chunks
from app.rag.retriever import HybridDatasheetRetriever
from app.schemas import HealthResponse, MetricsResponse, QueryRequest, RAGResponse

# Prevent stale local HF_TOKENs from blocking public model downloads
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("techspec_api")


class MetricsTracker:
    """Thread-safe in-memory observability store for /metrics."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.total_requests = 0
        self.answered_requests = 0
        self.refused_requests = 0
        self.error_requests = 0
        self._sum_retrieval_ms = 0.0
        self._sum_generation_ms = 0.0
        self._sum_total_ms = 0.0

    def record_success(self, resp: RAGResponse) -> None:
        with self._lock:
            self.total_requests += 1
            if resp.refused:
                self.refused_requests += 1
            else:
                self.answered_requests += 1
            self._sum_retrieval_ms += resp.retrieval_ms
            self._sum_generation_ms += resp.generation_ms
            self._sum_total_ms += resp.total_ms

    def record_error(self) -> None:
        with self._lock:
            self.total_requests += 1
            self.error_requests += 1

    def snapshot(self) -> MetricsResponse:
        with self._lock:
            completed = self.answered_requests + self.refused_requests
            refusal_rate = (
                round(100.0 * self.refused_requests / completed, 1)
                if completed > 0
                else 0.0
            )
            return MetricsResponse(
                total_requests=self.total_requests,
                answered_requests=self.answered_requests,
                refused_requests=self.refused_requests,
                error_requests=self.error_requests,
                refusal_rate_pct=refusal_rate,
                avg_retrieval_ms=round(self._sum_retrieval_ms / completed, 1) if completed else 0.0,
                avg_generation_ms=round(self._sum_generation_ms / completed, 1) if completed else 0.0,
                avg_total_ms=round(self._sum_total_ms / completed, 1) if completed else 0.0,
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.metrics = MetricsTracker()
    app.state.retriever = None
    app.state.pipeline = None
    app.state.chunk_count = 0

    try:
        if settings.skip_startup_init:
            logger.info("SKIP_STARTUP_INIT=1 set; skipping index & model startup.")
        else:
            logger.info("Loading datasheet chunks...")
            chunks = (
                load_chunks(settings.chunks_path)
                if settings.chunks_path.exists()
                else ingest_directory(settings.pdf_dir, settings.chunks_path)
            )
            retriever = HybridDatasheetRetriever(
                chunks=chunks,
                qdrant_path=settings.qdrant_path,
                collection_name=settings.collection_name,
                embed_model_name=settings.embed_model,
                reranker_model_name=settings.reranker_model,
            )
            retriever.build_vector_index(recreate=False)

            pipeline = GroundedRAGPipeline(
                retriever=retriever,
                model_name=settings.gemini_model,
                min_rerank_score=settings.min_rerank_score,
            )

            app.state.retriever = retriever
            app.state.pipeline = pipeline
            app.state.chunk_count = len(chunks)
            logger.info("RAG service ready (%d chunks indexed).", len(chunks))
        yield
    finally:
        if getattr(app.state, "retriever", None) is not None:
            try:
                app.state.retriever.close()
                logger.info("Closed Qdrant local client cleanly.")
            except Exception as exc:
                logger.warning("Error closing Qdrant client: %s", exc)


app = FastAPI(
    title="TechSpec Hybrid RAG API",
    description="Hybrid BM25 + Qdrant + Cross-Encoder RAG over engineering datasheets with strict citations and refusal guardrails.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_pipeline(request: Request) -> GroundedRAGPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized.")
    return pipeline


def get_chunk_count(request: Request) -> int:
    return int(getattr(request.app.state, "chunk_count", 0))


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health_check(
    request: Request,
    chunk_count: int = Depends(get_chunk_count),
) -> HealthResponse:
    settings = request.app.state.settings
    ready = getattr(request.app.state, "pipeline", None) is not None or chunk_count > 0
    return HealthResponse(
        status="ok" if ready else "initializing",
        indexed_chunks=chunk_count,
        model_name=settings.gemini_model,
        min_rerank_score=settings.min_rerank_score,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Ops"])
def get_metrics(request: Request) -> MetricsResponse:
    return request.app.state.metrics.snapshot()


@app.post("/query", response_model=RAGResponse, tags=["RAG"])
def query_datasheets(
    payload: QueryRequest,
    request: Request,
    pipeline: GroundedRAGPipeline = Depends(get_pipeline),
) -> RAGResponse:
    try:
        result = pipeline.answer(
            query=payload.query,
            top_k=payload.top_k,
            mode=payload.mode,
        )
        request.app.state.metrics.record_success(result)
        logger.info(
            "query_done | mode=%s | refused=%s | top_ce=%s | total_ms=%.1f",
            payload.mode,
            result.refused,
            result.top_rerank_score,
            result.total_ms,
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        request.app.state.metrics.record_error()
        logger.exception("Error processing query: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"RAG pipeline error: {exc}",
        ) from exc
