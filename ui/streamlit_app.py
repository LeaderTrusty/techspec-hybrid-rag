from __future__ import annotations

import os
import sys
from pathlib import Path
from threading import Lock
from typing import Any

# Keep public Hugging Face downloads clean and limit tokenizer thread contention
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Allow importing `app.*` when Streamlit launches from `ui/streamlit_app.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests
import streamlit as st

st.set_page_config(
    page_title="TechSpec Hybrid RAG — Engineering Datasheet Assistant",
    page_icon="🔧",
    layout="wide",
)


def _get_setting(key: str, default: str = "") -> str:
    """Read config from environment variables first, then Streamlit Secrets."""
    env_val = os.getenv(key)
    if env_val is not None and str(env_val).strip() != "":
        return str(env_val).strip()
    try:
        if key in st.secrets:
            secret_val = st.secrets[key]
            if secret_val is not None and str(secret_val).strip() != "":
                return str(secret_val).strip()
    except Exception:
        pass
    return default


API_URL = _get_setting("API_URL", "").rstrip("/")
USE_FASTAPI = bool(API_URL)
GEMINI_MODEL = _get_setting("GEMINI_MODEL", "gemini-3.5-flash-lite")
MIN_RERANK_SCORE = float(_get_setting("MIN_RERANK_SCORE", "1.5"))

PRESET_QUERIES = {
    "Select a sample engineering question...": "",
    "I2C Addresses & Supply Voltage (Multi-Page)": "What are the BME280 I2C 7-bit slave addresses and VDD supply voltage range?",
    "Humidity Oversampling Register (Table Lookup)": "Which register address and register name control humidity oversampling osrs_h?",
    "Pb-Free Reflow Soldering Profile (Cross-Doc)": "What is the peak Pb-free reflow soldering temperature and duration for BME280?",
    "Soft Reset Register & Hex Code": "Which register address performs a soft reset on the BME280 and what hex value must be written to it?",
    "[Trap] Out-of-Domain MCU (Should Refuse)": "What is the maximum CAN bus baud rate of the STM32F407 microcontroller?",
    "[Trap] Non-Existent BME280 Wi-Fi (Should Refuse)": "What is the BME280 Wi-Fi 2.4 GHz transmit power in dBm?",
}


class LocalMetricsTracker:
    """Lightweight in-process telemetry mirror of FastAPI's /metrics endpoint."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.total_requests = 0
        self.answered_requests = 0
        self.refused_requests = 0
        self._sum_retrieval_ms = 0.0
        self._sum_generation_ms = 0.0
        self._sum_total_ms = 0.0

    def record(self, data: dict[str, Any]) -> None:
        with self._lock:
            self.total_requests += 1
            if data.get("refused"):
                self.refused_requests += 1
            else:
                self.answered_requests += 1
            self._sum_retrieval_ms += float(data.get("retrieval_ms", 0.0))
            self._sum_generation_ms += float(data.get("generation_ms", 0.0))
            self._sum_total_ms += float(data.get("total_ms", 0.0))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            n = self.total_requests
            return {
                "total_requests": n,
                "answered_requests": self.answered_requests,
                "refused_requests": self.refused_requests,
                "refusal_rate_pct": round(100.0 * self.refused_requests / n, 1) if n else 0.0,
                "avg_retrieval_ms": round(self._sum_retrieval_ms / n, 1) if n else 0.0,
                "avg_generation_ms": round(self._sum_generation_ms / n, 1) if n else 0.0,
                "avg_total_ms": round(self._sum_total_ms / n, 1) if n else 0.0,
            }


@st.cache_resource
def get_local_metrics() -> LocalMetricsTracker:
    return LocalMetricsTracker()


@st.cache_resource(show_spinner="Loading Hybrid RAG models and Qdrant index (first start takes ~45s)...")
def load_direct_pipeline(api_key: str, model_name: str, min_rerank_score: float):
    import torch

    from app.rag.generator import GroundedRAGPipeline
    from app.rag.ingest import ingest_directory, load_chunks
    from app.rag.retriever import HybridDatasheetRetriever

    # Keep CPU thread footprint modest on shared cloud containers
    torch.set_num_threads(2)

    chunks_path = REPO_ROOT / "storage" / "chunks.jsonl"
    pdf_dir = REPO_ROOT / "data" / "datasheets"
    if chunks_path.exists():
        chunks = load_chunks(chunks_path)
    else:
        chunks = ingest_directory(pdf_dir=pdf_dir, out_jsonl=Path("/tmp/techspec_chunks.jsonl"))

    retriever = HybridDatasheetRetriever(
        chunks=chunks,
        qdrant_path=Path("/tmp/techspec_qdrant"),
    )
    retriever.build_vector_index(recreate=False)

    pipeline = GroundedRAGPipeline(
        retriever=retriever,
        model_name=model_name,
        min_rerank_score=min_rerank_score,
        api_key=api_key,
    )
    return pipeline, len(chunks)


# --- Initialize Runtime Backend & Sidebar Status ---
st.sidebar.header("⚙️ RAG Controls & Telemetry")

direct_pipeline = None
local_metrics = get_local_metrics()

if USE_FASTAPI:
    st.sidebar.caption(f"Runtime: **FastAPI Microservice** (`{API_URL}`)")
    try:
        h_resp = requests.get(f"{API_URL}/health", timeout=4)
        health = h_resp.json() if h_resp.ok else None
    except Exception:
        health = None

    if health and health.get("status") == "ok":
        st.sidebar.success(
            f"API Online • {health['indexed_chunks']} chunks\n\n"
            f"Model: `{health['model_name']}`"
        )
    else:
        st.sidebar.error(f"FastAPI backend unreachable at `{API_URL}`")
else:
    st.sidebar.caption("Runtime: **Direct In-Process** (Cloud Mode)")
    gemini_key = _get_setting("GEMINI_API_KEY", "")
    if not gemini_key:
        st.sidebar.error("Missing `GEMINI_API_KEY` in Streamlit Secrets.")
        st.error(
            "Add `GEMINI_API_KEY` in your Streamlit Community Cloud app **Settings → Secrets** to enable live queries."
        )
        st.stop()

    try:
        direct_pipeline, indexed_chunks = load_direct_pipeline(
            api_key=gemini_key,
            model_name=GEMINI_MODEL,
            min_rerank_score=MIN_RERANK_SCORE,
        )
        st.sidebar.success(
            f"Pipeline Ready • {indexed_chunks} chunks\n\n"
            f"Model: `{GEMINI_MODEL}` • Gate: `{MIN_RERANK_SCORE:+.1f}`"
        )
    except Exception as exc:
        st.sidebar.error("Failed to initialize RAG pipeline.")
        st.exception(exc)
        st.stop()

mode = st.sidebar.selectbox(
    "Retrieval Mode",
    options=["hybrid_rerank", "hybrid", "vector", "bm25"],
    index=0,
    help="Compare Hybrid + Cross-Encoder Reranking against single-stage baselines.",
)
top_k = st.sidebar.slider("Top-K Context Chunks", min_value=1, max_value=8, value=5)

st.sidebar.divider()
st.sidebar.subheader("📊 Live Telemetry")

metrics_data = None
if USE_FASTAPI:
    try:
        m_resp = requests.get(f"{API_URL}/metrics", timeout=4)
        if m_resp.ok:
            metrics_data = m_resp.json()
    except Exception:
        metrics_data = None
else:
    metrics_data = local_metrics.snapshot()

if metrics_data:
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Total Queries", metrics_data["total_requests"])
    c2.metric("Refusal Rate", f"{metrics_data['refusal_rate_pct']:.1f}%")
    st.sidebar.caption(
        f"Avg Latency: {metrics_data['avg_total_ms']:.0f} ms "
        f"(Ret: {metrics_data['avg_retrieval_ms']:.0f} ms | Gen: {metrics_data['avg_generation_ms']:.0f} ms)"
    )
else:
    st.sidebar.caption("Metrics unavailable.")

# --- Main Workspace ---
st.title("🔧 TechSpec Hybrid RAG — Hardware Datasheet Assistant")
st.markdown(
    "Ask hardware, register, electrical-limit, or soldering questions across indexed **Bosch Sensortec BME280** PDFs. "
    "Powered by **BM25 + Qdrant + Cross-Encoder Reranking + Gemini** with verified page citations and two-layer refusal guardrails."
)

preset_choice = st.selectbox("Quick Engineering Test Prompts", list(PRESET_QUERIES.keys()))
default_q = PRESET_QUERIES[preset_choice]

with st.form("rag_query_form", clear_on_submit=False):
    user_query = st.text_input(
        "Enter your technical specification question:",
        value=default_q,
        placeholder="e.g., What is the BME280 chip_id register address and value?",
    )
    submitted = st.form_submit_button("🔍 Search & Verify Specs", type="primary")

if submitted:
    cleaned_query = user_query.strip()
    if len(cleaned_query) < 3:
        st.warning("Please enter a question with at least 3 characters.")
    else:
        with st.spinner("Running Hybrid Search, Cross-Encoder Reranking, and Grounded Synthesis..."):
            try:
                if USE_FASTAPI:
                    resp = requests.post(
                        f"{API_URL}/query",
                        json={"query": cleaned_query, "top_k": top_k, "mode": mode},
                        timeout=90,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                else:
                    rag_result = direct_pipeline.answer(
                        query=cleaned_query,
                        top_k=top_k,
                        mode=mode,
                    )
                    data = rag_result.model_dump()
                    local_metrics.record(data)
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                st.stop()

        # Status Banner
        if data["refused"]:
            st.warning(
                f"🛡️ **Guardrail Refusal Triggered** — Reason: `{data['refusal_reason']}`"
            )
        else:
            st.success("✅ **Grounded Answer Verified**")

        # Answer Box
        st.markdown("### Answer")
        st.markdown(data["answer"])

        # Telemetry Row
        k1, k2, k3, k4 = st.columns(4)
        score_val = (
            f"{data['top_rerank_score']:+.3f}"
            if data.get("top_rerank_score") is not None
            else "N/A"
        )
        k1.metric("Top Cross-Encoder Score", score_val)
        k2.metric("Retrieval Latency", f"{data['retrieval_ms']:.0f} ms")
        k3.metric("LLM Latency", f"{data['generation_ms']:.0f} ms")
        k4.metric("Verified Citations", len(data["citations"]))

        # Verified Citations Panel
        if data["citations"]:
            st.markdown("### 📄 Verified Source Citations")
            for idx, cite in enumerate(data["citations"], start=1):
                label = (
                    f"[{idx}] {cite['doc_name']} — Page {cite['page']} "
                    f"({cite['section']}) | chunk: `{cite['chunk_id']}`"
                )
                with st.expander(label, expanded=(idx == 1)):
                    st.markdown(f"> {cite['quote']}")
                    if cite.get("rerank_score") is not None:
                        st.caption(f"Cross-Encoder relevance logit: `{cite['rerank_score']:+.3f}`")
