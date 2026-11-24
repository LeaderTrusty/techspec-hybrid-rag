from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from app.rag.ingest import load_chunks
from app.rag.retriever import HybridDatasheetRetriever, RetrievalMode

MODES: list[tuple[str, RetrievalMode]] = [
    ("1. BM25-Only (Lexical)", "bm25"),
    ("2. Vector-Only (Dense BGE)", "vector"),
    ("3. Hybrid (BM25 + Vector RRF)", "hybrid"),
    ("4. Hybrid RRF + Cross-Encoder", "hybrid_rerank"),
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def chunk_matches_item(chunk_text: str, doc_name: str, item: dict[str, Any]) -> bool:
    if item["expected_docs"] and doc_name not in item["expected_docs"]:
        return False
    norm_text = _normalize(chunk_text)
    for group in item["required_groups"]:
        if not any(_normalize(alias) in norm_text for alias in group):
            return False
    return True


def evaluate_retrieval(
    retriever: HybridDatasheetRetriever,
    dataset_path: Path = Path("eval/golden_dataset.json"),
    min_rerank_score: float = 0.0,
    candidate_k: int = 15,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    in_scope = [x for x in dataset if x["answerable"]]
    traps = [x for x in dataset if not x["answerable"]]

    rows = []
    per_mode_misses: dict[str, list[str]] = {}

    for label, mode in MODES:
        hit1 = hit3 = hit5 = 0
        rr_sum = 0.0
        latencies = []
        misses = []

        for item in in_scope:
            t0 = time.perf_counter()
            results = retriever.retrieve(
                item["query"], top_k=5, candidate_k=candidate_k, mode=mode
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)

            first_hit_rank = None
            for rank, res in enumerate(results, start=1):
                if chunk_matches_item(res.chunk.text, res.chunk.doc_name, item):
                    first_hit_rank = rank
                    break

            if first_hit_rank is not None:
                if first_hit_rank <= 1:
                    hit1 += 1
                if first_hit_rank <= 3:
                    hit3 += 1
                if first_hit_rank <= 5:
                    hit5 += 1
                rr_sum += 1.0 / first_hit_rank
            else:
                misses.append(item["id"])

        n = len(in_scope)
        per_mode_misses[mode] = misses
        rows.append(
            {
                "Retrieval Mode": label,
                "mode_key": mode,
                "Hit@1 (%)": round(100.0 * hit1 / n, 1),
                "Hit@3 (%)": round(100.0 * hit3 / n, 1),
                "Hit@5 (%)": round(100.0 * hit5 / n, 1),
                "MRR@5": round(rr_sum / n, 4),
                "Avg Latency (ms)": round(sum(latencies) / len(latencies), 1),
            }
        )

    bench_df = pd.DataFrame(rows)

    # --- Guardrail Evaluation (Pre-LLM Cross-Encoder Confidence Gate) ---
    score_records = []
    for item in dataset:
        hits = retriever.retrieve(
            item["query"], top_k=1, candidate_k=candidate_k, mode="hybrid_rerank"
        )
        top_ce = float(hits[0].rerank_score) if hits and hits[0].rerank_score is not None else -99.0
        predicted_refuse = top_ce < min_rerank_score
        expected_refuse = not item["answerable"]
        score_records.append(
            {
                "id": item["id"],
                "category": item["category"],
                "answerable": item["answerable"],
                "top_ce_score": round(top_ce, 3),
                "predicted_refuse": predicted_refuse,
                "correct": predicted_refuse == expected_refuse,
            }
        )

    in_scope_scores = [r["top_ce_score"] for r in score_records if r["answerable"]]
    trap_scores = [r["top_ce_score"] for r in score_records if not r["answerable"]]

    true_refusals = sum(1 for r in score_records if (not r["answerable"]) and r["predicted_refuse"])
    false_refusals = sum(1 for r in score_records if r["answerable"] and r["predicted_refuse"])
    overall_guard_acc = sum(1 for r in score_records if r["correct"]) / len(score_records)

    guardrail_summary = {
        "threshold": min_rerank_score,
        "in_scope_count": len(in_scope),
        "trap_count": len(traps),
        "true_refusal_rate_pct": round(100.0 * true_refusals / len(traps), 1),
        "false_refusal_rate_pct": round(100.0 * false_refusals / len(in_scope), 1),
        "overall_guardrail_accuracy_pct": round(100.0 * overall_guard_acc, 1),
        "min_in_scope_ce_score": min(in_scope_scores),
        "max_trap_ce_score": max(trap_scores),
        "separation_margin": round(min(in_scope_scores) - max(trap_scores), 3),
        "per_item_scores": score_records,
        "per_mode_misses": per_mode_misses,
    }

    bench_df.to_csv("eval/retrieval_benchmark.csv", index=False)
    Path("eval/eval_report.json").write_text(
        json.dumps(
            {"retrieval": rows, "guardrail": guardrail_summary},
            indent=2,
        ),
        encoding="utf-8",
    )
    return bench_df, guardrail_summary


if __name__ == "__main__":
    chunks = load_chunks(Path("storage/chunks.jsonl"))
    ret = HybridDatasheetRetriever(chunks=chunks)
    ret.build_vector_index(recreate=False)
    try:
        df_res, guard = evaluate_retrieval(ret)
        print(df_res.drop(columns=["mode_key"]).to_string(index=False))
        print("\nGuardrail Summary:", json.dumps({k: v for k, v in guard.items() if k != "per_item_scores"}, indent=2))
    finally:
        ret.close()
