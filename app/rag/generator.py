from __future__ import annotations

import os
import random
import time
from typing import List

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.rag.retriever import HybridDatasheetRetriever, RetrievalMode, RetrievedChunk
from app.schemas import LLMGroundedOutput, RAGResponse, VerifiedCitation

SYSTEM_PROMPT = """You are a strict hardware & engineering datasheet assistant.
Answer the user's question using ONLY the retrieved datasheet chunks provided below.

Hard rules:
1. Never use outside memory or guess electrical limits, register addresses, or specs.
2. Include exact units, tolerances, hex addresses, and pin names exactly as written in the chunks.
3. Add inline citations like [bosch_bme280_datasheet.pdf, p. 28] after every technical claim.
4. Populate `citations` using ONLY exact `chunk_id` values present in the context.
5. If the chunks do not contain the requested specification (or only partially mention the component without the asked parameter), set `is_answerable` to false and state clearly that the provided datasheets do not contain that specification.
"""

RETRYABLE_STATUS_CODES = {429, 500, 503, 504}


def _format_context(retrieved: List[RetrievedChunk]) -> str:
    blocks = []
    for res in retrieved:
        c = res.chunk
        blocks.append(
            f"--- CHUNK_ID: {c.chunk_id} | DOC: {c.doc_name} | PAGE: {c.page} | SECTION: {c.section} ---\n"
            f"{c.text}"
        )
    return "\n\n".join(blocks)


class GroundedRAGPipeline:
    def __init__(
        self,
        retriever: HybridDatasheetRetriever,
        model_name: str = "gemini-3.5-flash-lite",
        min_rerank_score: float = 0.0,
        max_retries: int = 5,
        base_backoff_s: float = 2.0,
        api_key: str | None = None,
    ) -> None:
        self.retriever = retriever
        self.model_name = model_name
        self.min_rerank_score = min_rerank_score
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("GEMINI_API_KEY environment variable is required.")
        self.client = genai.Client(api_key=resolved_key)

    def _generate_with_retry(self, user_prompt: str) -> str:
        """Call Gemini API with exponential backoff + jitter on transient 429/5xx errors."""
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw_resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=LLMGroundedOutput,
                    ),
                )
                return raw_resp.text
            except genai_errors.APIError as exc:
                last_exc = exc
                status_code = getattr(exc, "code", None)
                if status_code not in RETRYABLE_STATUS_CODES or attempt == self.max_retries:
                    raise
                delay = self.base_backoff_s * (2 ** (attempt - 1)) + random.uniform(0.2, 1.0)
                print(
                    f"[RETRY {attempt}/{self.max_retries}] Gemini returned HTTP {status_code}. "
                    f"Waiting {delay:.1f}s..."
                )
                time.sleep(delay)

        raise RuntimeError(f"Gemini generation failed after {self.max_retries} attempts: {last_exc}")

    def answer(
        self,
        query: str,
        top_k: int = 5,
        mode: RetrievalMode = "hybrid_rerank",
    ) -> RAGResponse:
        t0 = time.perf_counter()

        # 1. Hybrid Retrieval + Cross-Encoder Reranking
        retrieved = self.retriever.retrieve(query, top_k=top_k, mode=mode)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        top_score = (
            retrieved[0].rerank_score
            if retrieved and retrieved[0].rerank_score is not None
            else None
        )

        # 2. Layer 1 Guardrail: Deterministic Pre-LLM Confidence Gate
        if not retrieved or (top_score is not None and top_score < self.min_rerank_score):
            total_ms = (time.perf_counter() - t0) * 1000.0
            return RAGResponse(
                query=query,
                answer=(
                    "I cannot answer this from the loaded engineering datasheets. "
                    "No sufficiently relevant passages were found in the indexed documents."
                ),
                refused=True,
                refusal_reason=(
                    f"pre_llm_low_confidence (top_ce_score={top_score:.3f} < {self.min_rerank_score})"
                    if top_score is not None
                    else "no_chunks_retrieved"
                ),
                top_rerank_score=top_score,
                citations=[],
                retrieval_ms=round(retrieval_ms, 1),
                generation_ms=0.0,
                total_ms=round(total_ms, 1),
                model_used="guardrail_bypass",
            )

        # 3. Grounded LLM Synthesis with Strict Structured Output + Retries
        t_gen = time.perf_counter()
        context_str = _format_context(retrieved)
        user_prompt = f"DATASHEET CONTEXT:\n{context_str}\n\nUSER QUESTION:\n{query}"

        raw_json = self._generate_with_retry(user_prompt)
        generation_ms = (time.perf_counter() - t_gen) * 1000.0

        parsed = LLMGroundedOutput.model_validate_json(raw_json)

        # 4. Deterministic Citation Verification (strip any chunk_id not in retrieved top-k)
        chunk_lookup = {r.chunk.chunk_id: r for r in retrieved}
        verified_citations: list[VerifiedCitation] = []
        seen_ids: set[str] = set()

        for cite in parsed.citations:
            if cite.chunk_id in chunk_lookup and cite.chunk_id not in seen_ids:
                seen_ids.add(cite.chunk_id)
                r = chunk_lookup[cite.chunk_id]
                verified_citations.append(
                    VerifiedCitation(
                        chunk_id=r.chunk.chunk_id,
                        doc_name=r.chunk.doc_name,
                        page=r.chunk.page,
                        section=r.chunk.section,
                        quote=cite.quote,
                        rerank_score=r.rerank_score,
                    )
                )

        # 5. Layer 2 Guardrail: Enforce refusal if LLM flags unanswerable or cites zero valid chunks
        refused = (not parsed.is_answerable) or (len(verified_citations) == 0)
        refusal_reason = None
        if not parsed.is_answerable:
            refusal_reason = "llm_insufficient_context"
        elif len(verified_citations) == 0:
            refusal_reason = "no_verified_citations"

        total_ms = (time.perf_counter() - t0) * 1000.0
        return RAGResponse(
            query=query,
            answer=parsed.answer,
            refused=refused,
            refusal_reason=refusal_reason,
            top_rerank_score=top_score,
            citations=verified_citations if not refused else [],
            retrieval_ms=round(retrieval_ms, 1),
            generation_ms=round(generation_ms, 1),
            total_ms=round(total_ms, 1),
            model_used=self.model_name,
        )
