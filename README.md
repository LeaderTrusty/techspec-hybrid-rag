# 🔧 TechSpec Hybrid RAG — Hardware Datasheet Assistant

A containerized **Hybrid RAG microservice** for querying technical electronics datasheets (**Bosch Sensortec BME280 Datasheet & Handling/Soldering Guide**) with table-preserving PDF parsing, inline page citations, and a two-layer anti-hallucination refusal guardrail.

---

## 📌 Why Naive Vector RAG Fails on Engineering Datasheets
Standard dense-embedding RAG pipelines struggle on hardware documentation for three reasons:
1. **Exact Identifiers Get Blurred:** Dense vectors often rank conceptual prose above exact hex register addresses (`0xF2`, `0xD0`, `0x76`/`0x77`) or pin symbols (`osrs_h`, `VDDIO`).
2. **Broken Parameter Tables:** Fixed-character chunkers split PDF tables mid-row, separating electrical limits from their column headers and units.
3. **Silent Hallucination on Out-of-Scope Specs:** Vector search always returns a nearest neighbor—even when asked about a completely different microcontroller or unsupported wireless feature.

---

## ⚙️ System Architecture
* **Table-Aware PDF Ingestion (`app/rag/ingest.py`):** Uses `pymupdf4llm` (`page_chunks=True`) to preserve 1-based page numbers, section headers, and intact Markdown parameter tables—repeating header rows whenever oversized tables split.
* **Hybrid Retrieval + Reranking (`app/rag/retriever.py`):** Combines a custom technical-token **BM25** lexical index (preserving hex codes, decimals, and register underscores) with **Qdrant + `BAAI/bge-small-en-v1.5`** dense vectors via **Reciprocal Rank Fusion (RRF, $k=60$)**, followed by **`cross-encoder/ms-marco-MiniLM-L-6-v2`** reranking.
* **Two-Layer Anti-Hallucination Guardrail (`app/rag/generator.py`):**
  1. **Layer 1 (Deterministic Pre-LLM Gate):** Rejects queries whose top Cross-Encoder logit falls below `MIN_RERANK_SCORE`—blocking out-of-scope traps with zero LLM API tokens.
  2. **Layer 2 (Structured Schema + Citation Verification):** Uses `google-genai` (`gemini-3.5-flash-lite`) with exponential backoff retries on `429`/`503` spikes, strict Pydantic JSON output, and deterministic verification that every cited `chunk_id` exists in the retrieved top-$k$ context.
* **Production Packaging (`app/main.py`, `Dockerfile`, `docker-compose.yml`):** Exposes `/query`, `/health`, and `/metrics` via FastAPI (`lifespan` management) paired with a decoupled Streamlit UI and offline `pytest` suite.

---

## 📊 Automated Evaluation Benchmark (20-Question Golden Suite)
Evaluated across **15 real BME280 hardware, register, and soldering questions** plus **5 unanswerable hallucination traps** (`eval/golden_dataset.json`):

| Retrieval Mode | Hit@1 (%) | Hit@3 (%) | Hit@5 (%) | MRR@5 | Avg Latency (ms)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. BM25-Only (Lexical) | 73.3 | 93.3 | 93.3 | 0.8333 | 7.0 |
| 2. Vector-Only (Dense BGE) | 46.7 | 80.0 | 93.3 | 0.6556 | 209.6 |
| 3. Hybrid (BM25 + Vector RRF) | 53.3 | 86.7 | 86.7 | 0.7000 | 126.4 |
| **4. Hybrid RRF + Cross-Encoder** | **80.0** | **93.3** | **93.3** | **0.8556** | **6,949.8** |

*\*Measured on Google Colab shared CPU (`candidate_k=15`).*

### Key Benchmark Takeaways
* **+33.3% Hit@1 Lift Over Vector-Only:** Pure dense search dropped to **46.7% Hit@1** on exact register and pin specs, while Hybrid RRF + Cross-Encoder reached **80.0% Hit@1** and the highest **MRR@5 (0.8556)**.
* **Multi-Chunk Synthesis (`Q10`):** Single-chunk Hit@5 missed `Q10` across all modes because the I2C (`3.4 MHz`) and SPI (`10 MHz`) clock limits appear in separate chunks; passing `top_k=5` chunks into synthesis resolved both specs cleanly.
* **Two-Layer Guardrail Defense in Depth:**
  * **Layer 1 Pre-LLM Gate (`threshold = 0.0`):** **80.0%** trap refusal (4/5 blocked at zero token cost), **0.0%** false refusal rate (15/15 in-scope passed), **95.0%** overall gate accuracy.
  * **Clean Logit Separation Margin:** Lowest in-scope score **`+2.614`** vs. highest trap score **`+0.878`** (**`+1.736`** logit gap—allowing calibration to `MIN_RERANK_SCORE=1.5` for 100% pre-LLM separation).
  * **Layer 2 Grounded LLM Audit (`gemini-3.5-flash-lite`):** Caught the single subtle in-domain BME280 trap that crossed `0.0`, achieving **15/15 (100.0%)** in-scope spec & citation accuracy, **5/5 (100.0%)** trap refusal accuracy, and **20/20 (100.0%)** overall end-to-end pass rate.

![Retrieval Benchmark & Guardrail Separation](images/01_retrieval_benchmark.png)

---

## 📁 Repository Structure
```text
techspec-hybrid-rag/
├── app/
│   ├── main.py              # FastAPI service (/query, /health, /metrics)
│   ├── config.py            # Environment-driven Settings
│   ├── schemas.py           # Pydantic API, citation & telemetry contracts
│   └── rag/
│       ├── ingest.py        # Table-aware PDF parser & page-tracked chunking
│       ├── retriever.py     # Technical BM25 + Qdrant Vector + RRF + Cross-Encoder
│       └── generator.py     # Gemini 3.5 Flash-Lite + retry + 2-layer guardrail
├── ui/
│   └── streamlit_app.py     # Thin Streamlit frontend calling FastAPI over HTTP
├── eval/
│   ├── golden_dataset.json  # 15 BME280 spec questions + 5 unanswerable traps
│   ├── run_eval.py          # 4-mode retrieval & guardrail benchmark script
│   ├── retrieval_benchmark.csv
│   ├── eval_report.json
│   └── e2e_results.csv
├── tests/
│   └── test_api.py          # Offline pytest suite (API, metrics & guardrail)
├── scripts/
│   └── download_datasheets.py
├── images/
│   └── 01_retrieval_benchmark.png
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .dockerignore
└── .gitignore
```

---

## 🚀 Quickstart with Docker Compose
```bash
git clone [https://github.com/LeaderTrusty/techspec-hybrid-rag.git](https://github.com/LeaderTrusty/techspec-hybrid-rag.git)
cd techspec-hybrid-rag

# 1. Configure your free Gemini API key
cp .env.example .env
# Edit .env and paste your GEMINI_API_KEY

# 2. Download the Bosch Sensortec PDFs (if not already in data/datasheets/)
python scripts/download_datasheets.py

# 3. Build and launch FastAPI (:8000) + Streamlit UI (:8501)
docker compose up --build
```

* **Streamlit UI:** `http://localhost:8501`
* **FastAPI Swagger Docs:** `http://localhost:8000/docs`
* **Health Check:** `http://localhost:8000/health`
* **Live Telemetry:** `http://localhost:8000/metrics`

---

## 🧪 Running Automated Tests
```bash
pip install -r requirements.txt
PYTHONPATH=. pytest -v
```

---

## 📂 Data Sources & Attribution
* **Bosch Sensortec BME280 Datasheet (`BST-BME280-DS002`):** Official public hardware specification, Bosch Sensortec GmbH.
* **Bosch Sensortec BME280 Handling, Soldering & Mounting Instructions (`BST-BME280-HS006`):** Official public assembly specification, Bosch Sensortec GmbH.
