from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class LLMCitation(BaseModel):
    chunk_id: str = Field(description="Exact chunk_id from the provided context.")
    quote: str = Field(description="Short verbatim phrase or parameter row supporting the claim.")


class LLMGroundedOutput(BaseModel):
    is_answerable: bool = Field(
        description="True ONLY if the provided chunks explicitly contain the answer."
    )
    answer: str = Field(
        description=(
            "Concise technical answer with inline citations formatted as [doc_name, p. X]. "
            "If is_answerable is False, state clearly what is missing from the datasheets."
        )
    )
    citations: List[LLMCitation] = Field(
        default_factory=list,
        description="List of exact chunk_ids and supporting quotes used in the answer.",
    )


class VerifiedCitation(BaseModel):
    chunk_id: str
    doc_name: str
    page: int
    section: str
    quote: str
    rerank_score: Optional[float] = None


class RAGResponse(BaseModel):
    query: str
    answer: str
    refused: bool
    refusal_reason: Optional[str] = None
    top_rerank_score: Optional[float] = None
    citations: List[VerifiedCitation] = Field(default_factory=list)
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    model_used: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="Engineering specification question.")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of top chunks passed to synthesis.")
    mode: Literal["vector", "bm25", "hybrid", "hybrid_rerank"] = Field(
        default="hybrid_rerank",
        description="Retrieval strategy to execute.",
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "initializing"]
    indexed_chunks: int
    model_name: str
    min_rerank_score: float


class MetricsResponse(BaseModel):
    total_requests: int
    answered_requests: int
    refused_requests: int
    error_requests: int
    refusal_rate_pct: float
    avg_retrieval_ms: float
    avg_generation_ms: float
    avg_total_ms: float
