from __future__ import annotations

import os

os.environ["SKIP_STARTUP_INIT"] = "1"

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_chunk_count, get_pipeline
from app.rag.generator import GroundedRAGPipeline
from app.rag.ingest import DocumentChunk
from app.rag.retriever import RetrievedChunk, technical_tokenize
from app.schemas import RAGResponse, VerifiedCitation


class FakePipeline:
    def answer(self, query: str, top_k: int = 5, mode: str = "hybrid_rerank") -> RAGResponse:
        if "stm32" in query.lower() or "wi-fi" in query.lower():
            return RAGResponse(
                query=query,
                answer="I cannot answer this from the loaded engineering datasheets.",
                refused=True,
                refusal_reason="pre_llm_low_confidence (top_ce_score=-9.500 < 0.0)",
                top_rerank_score=-9.5,
                citations=[],
                retrieval_ms=12.4,
                generation_ms=0.0,
                total_ms=12.5,
                model_used="guardrail_bypass",
            )

        return RAGResponse(
            query=query,
            answer="Register 0xF2 (ctrl_hum) controls humidity oversampling osrs_h [bosch_bme280_datasheet.pdf, p. 28].",
            refused=False,
            refusal_reason=None,
            top_rerank_score=5.232,
            citations=[
                VerifiedCitation(
                    chunk_id="bosch_bme280_datasheet_p028_c02",
                    doc_name="bosch_bme280_datasheet.pdf",
                    page=28,
                    section="Register 0xF2",
                    quote="Register 0xF2 | ctrl_hum | osrs_h[2:0]",
                    rerank_score=5.232,
                )
            ],
            retrieval_ms=25.0,
            generation_ms=410.0,
            total_ms=435.0,
            model_used="gemini-3.5-flash-lite",
        )


@pytest.fixture()
def client():
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    app.dependency_overrides[get_chunk_count] = lambda: 284
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_technical_tokenizer_preserves_hex_and_registers():
    tokens = technical_tokenize("Register 0xF2 ctrl_hum osrs_h VDD=1.71V")
    assert "0xf2" in tokens
    assert "ctrl_hum" in tokens
    assert "hum" in tokens
    assert "1.71v" in tokens


def test_health_endpoint(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["indexed_chunks"] == 284


def test_query_answered_with_verified_citations(client: TestClient):
    resp = client.post(
        "/query",
        json={"query": "Which register controls osrs_h?", "top_k": 5, "mode": "hybrid_rerank"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is False
    assert len(body["citations"]) == 1
    assert body["citations"][0]["page"] == 28


def test_query_refusal_and_metrics_tracking(client: TestClient):
    # 1 answered + 1 refused query -> 50% refusal rate
    r1 = client.post("/query", json={"query": "Which register controls osrs_h?"})
    r2 = client.post("/query", json={"query": "What is the STM32F407 CAN baud rate?"})
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r2.json()["refused"] is True
    assert r2.json()["citations"] == []

    m = client.get("/metrics").json()
    assert m["total_requests"] == 2
    assert m["answered_requests"] == 1
    assert m["refused_requests"] == 1
    assert m["refusal_rate_pct"] == 50.0


def test_query_validation_rejects_invalid_payload(client: TestClient):
    bad_short = client.post("/query", json={"query": "hi"})
    bad_k = client.post("/query", json={"query": "Valid BME280 query", "top_k": 50})
    assert bad_short.status_code == 422
    assert bad_k.status_code == 422


def test_layer1_guardrail_bypasses_llm_on_low_score():
    class LowScoreRetriever:
        def retrieve(self, query: str, top_k: int = 5, mode: str = "hybrid_rerank"):
            dummy_chunk = DocumentChunk(
                chunk_id="dummy_p001_c01",
                doc_name="bosch_bme280_datasheet.pdf",
                page=1,
                section="Intro",
                is_table=False,
                text="Unrelated sensor text",
            )
            return [RetrievedChunk(chunk=dummy_chunk, score=-8.4, rerank_score=-8.4)]

    # Bypass __init__ Gemini client setup to prove Layer 1 never calls the LLM
    pipeline = GroundedRAGPipeline.__new__(GroundedRAGPipeline)
    pipeline.retriever = LowScoreRetriever()
    pipeline.model_name = "gemini-3.5-flash-lite"
    pipeline.min_rerank_score = 0.0
    pipeline.client = None

    result = pipeline.answer("What is the STM32F407 CAN baud rate?")
    assert result.refused is True
    assert result.generation_ms == 0.0
    assert result.model_used == "guardrail_bypass"
