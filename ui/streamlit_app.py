import os
import requests
import streamlit as st

st.set_page_config(
    page_title="TechSpec Hybrid RAG — Engineering Datasheet Assistant",
    page_icon="🔧",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

PRESET_QUERIES = {
    "Select a sample engineering question...": "",
    "I2C Addresses & Supply Voltage (Multi-Page)": "What are the BME280 I2C 7-bit slave addresses and VDD supply voltage range?",
    "Humidity Oversampling Register (Table Lookup)": "Which register address and register name control humidity oversampling osrs_h?",
    "Pb-Free Reflow Soldering Profile (Cross-Doc)": "What is the peak Pb-free reflow soldering temperature and duration for BME280?",
    "Soft Reset Register & Hex Code": "Which register address performs a soft reset on the BME280 and what hex value must be written to it?",
    "[Trap] Out-of-Domain MCU (Should Refuse)": "What is the maximum CAN bus baud rate of the STM32F407 microcontroller?",
    "[Trap] Non-Existent BME280 Wi-Fi (Should Refuse)": "What is the BME280 Wi-Fi 2.4 GHz transmit power in dBm?",
}

# --- Sidebar: Backend Health, Controls & Live Metrics ---
st.sidebar.header("⚙️ RAG Controls & Telemetry")

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

mode = st.sidebar.selectbox(
    "Retrieval Mode",
    options=["hybrid_rerank", "hybrid", "vector", "bm25"],
    index=0,
    help="Compare Hybrid + Cross-Encoder Reranking against single-stage baselines.",
)
top_k = st.sidebar.slider("Top-K Context Chunks", min_value=1, max_value=8, value=5)

st.sidebar.divider()
st.sidebar.subheader("📊 Live API Metrics (`/metrics`)")
try:
    m_resp = requests.get(f"{API_URL}/metrics", timeout=4)
    if m_resp.ok:
        m = m_resp.json()
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Total Queries", m["total_requests"])
        c2.metric("Refusal Rate", f"{m['refusal_rate_pct']:.1f}%")
        st.sidebar.caption(
            f"Avg Latency: {m['avg_total_ms']:.0f} ms "
            f"(Ret: {m['avg_retrieval_ms']:.0f} ms | Gen: {m['avg_generation_ms']:.0f} ms)"
        )
except Exception:
    st.sidebar.caption("Metrics unavailable.")

# --- Main Workspace ---
st.title("🔧 TechSpec Hybrid RAG — Hardware Datasheet Assistant")
st.markdown(
    "Ask hardware, register, electrical-limit, or soldering questions across indexed **Bosch Sensortec BME280** PDFs. "
    "Powered by **FastAPI + BM25 + Qdrant + Cross-Encoder Reranking + Gemini** with strict citation verification and confidence-gated refusals."
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
    if len(user_query.strip()) < 3:
        st.warning("Please enter a question with at least 3 characters.")
    else:
        with st.spinner("Running Hybrid Search, Cross-Encoder Reranking, and Grounded Synthesis..."):
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={"query": user_query.strip(), "top_k": top_k, "mode": mode},
                    timeout=90,
                )
                resp.raise_for_status()
                data = resp.json()
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
