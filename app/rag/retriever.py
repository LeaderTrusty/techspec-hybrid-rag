from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, List

from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.rag.ingest import DocumentChunk

RetrievalMode = Literal["vector", "bm25", "hybrid", "hybrid_rerank"]

TECH_TOKEN_RE = re.compile(r"0x[0-9a-fA-F]+|[a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*")


def technical_tokenize(text: str) -> list[str]:
    raw_tokens = TECH_TOKEN_RE.findall(text.lower())
    tokens: list[str] = []
    for tok in raw_tokens:
        tokens.append(tok)
        if "_" in tok or "-" in tok:
            parts = [p for p in re.split(r"[_-]", tok) if len(p) >= 2]
            tokens.extend(parts)
    return tokens


class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class HybridDatasheetRetriever:
    def __init__(
        self,
        chunks: List[DocumentChunk],
        qdrant_path: str | Path = "storage/qdrant_db",
        collection_name: str = "techspec_datasheets",
        embed_model_name: str = "BAAI/bge-small-en-v1.5",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rrf_k: int = 60,
    ) -> None:
        self.chunks = chunks
        self.collection_name = collection_name
        self.rrf_k = rrf_k

        self.embedder = SentenceTransformer(embed_model_name)
        self.reranker = CrossEncoder(reranker_model_name)

        Path(qdrant_path).mkdir(parents=True, exist_ok=True)
        self.qdrant = QdrantClient(path=str(qdrant_path))

        tokenized_corpus = [technical_tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def close(self) -> None:
        self.qdrant.close()

    def build_vector_index(self, recreate: bool = True) -> None:
        if not recreate and self.qdrant.collection_exists(self.collection_name):
            info = self.qdrant.get_collection(self.collection_name)
            if info.points_count == len(self.chunks):
                print(f"[SKIP] Qdrant '{self.collection_name}' already indexed ({info.points_count} vectors).")
                return

        print(f"Embedding {len(self.chunks)} chunks...")
        vectors = self.embedder.encode(
            [c.text for c in self.chunks],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        if self.qdrant.collection_exists(self.collection_name):
            self.qdrant.delete_collection(self.collection_name)

        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vectors.shape[1],
                distance=models.Distance.COSINE,
            ),
        )

        points = [
            models.PointStruct(
                id=idx,
                vector=vectors[idx].tolist(),
                payload=self.chunks[idx].model_dump(),
            )
            for idx in range(len(self.chunks))
        ]
        self.qdrant.upsert(collection_name=self.collection_name, points=points)
        print(f"[OK] Indexed {len(points)} dense vectors into Qdrant ('{self.collection_name}').")

    def _search_bm25(self, query: str, top_k: int) -> list[tuple[int, float]]:
        tokens = technical_tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in ranked_indices[:top_k] if scores[i] > 0.0]

    def _search_vector(self, query: str, top_k: int) -> list[tuple[int, float]]:
        bge_query = f"Represent this sentence for searching relevant passages: {query}"
        q_vec = self.embedder.encode(bge_query, normalize_embeddings=True).tolist()
        hits = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=q_vec,
            limit=top_k,
            with_payload=False,
        ).points
        return [(int(h.id), float(h.score)) for h in hits]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        mode: RetrievalMode = "hybrid_rerank",
    ) -> List[RetrievedChunk]:
        if mode == "bm25":
            bm25_hits = self._search_bm25(query, top_k=top_k)
            return [
                RetrievedChunk(chunk=self.chunks[idx], score=score, bm25_rank=rank)
                for rank, (idx, score) in enumerate(bm25_hits, start=1)
            ]

        if mode == "vector":
            vec_hits = self._search_vector(query, top_k=top_k)
            return [
                RetrievedChunk(chunk=self.chunks[idx], score=score, vector_rank=rank)
                for rank, (idx, score) in enumerate(vec_hits, start=1)
            ]

        bm25_hits = self._search_bm25(query, top_k=candidate_k)
        vec_hits = self._search_vector(query, top_k=candidate_k)

        bm25_ranks = {idx: rank for rank, (idx, _) in enumerate(bm25_hits, start=1)}
        vec_ranks = {idx: rank for rank, (idx, _) in enumerate(vec_hits, start=1)}
        candidate_ids = set(bm25_ranks.keys()) | set(vec_ranks.keys())

        rrf_scores: dict[int, float] = {}
        for idx in candidate_ids:
            score = 0.0
            if idx in bm25_ranks:
                score += 1.0 / (self.rrf_k + bm25_ranks[idx])
            if idx in vec_ranks:
                score += 1.0 / (self.rrf_k + vec_ranks[idx])
            rrf_scores[idx] = score

        fused_ids = sorted(candidate_ids, key=lambda i: rrf_scores[i], reverse=True)[:candidate_k]

        if mode == "hybrid":
            return [
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=rrf_scores[idx],
                    bm25_rank=bm25_ranks.get(idx),
                    vector_rank=vec_ranks.get(idx),
                    rrf_score=rrf_scores[idx],
                )
                for idx in fused_ids[:top_k]
            ]

        if not fused_ids:
            return []

        pairs = [(query, self.chunks[idx].text) for idx in fused_ids]
        ce_scores = self.reranker.predict(pairs)

        reranked = sorted(
            zip(fused_ids, ce_scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:top_k]

        return [
            RetrievedChunk(
                chunk=self.chunks[idx],
                score=float(ce_score),
                bm25_rank=bm25_ranks.get(idx),
                vector_rank=vec_ranks.get(idx),
                rrf_score=rrf_scores[idx],
                rerank_score=float(ce_score),
            )
            for idx, ce_score in reranked
        ]
